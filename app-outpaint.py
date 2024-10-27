import gradio as gr
import torch
import devicetorch
import os
import webbrowser #for open_outputs_folder. webbrowser seemed most likely to work cross platform
import numpy as np
import gc

from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from controlnet_union import ControlNetModel_Union
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline

from huggingface_hub import hf_hub_download
from gradio_imageslider import ImageSlider
# from gradio import SelectData
from collections import deque
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw

DEVICE = devicetorch.get(torch)

MAX_GALLERY_IMAGES = 20
OUTPUT_DIR = "outputs"

pipe = None
gallery_images = deque(maxlen=MAX_GALLERY_IMAGES)
selected_image_index = None


def init(progress=gr.Progress()):
    global pipe

    if pipe is None:
        progress(0.1, desc="Starting model initialization")
        
        progress(0.2, desc="Loading ControlNet configuration")
        config_file = hf_hub_download(
            "xinsir/controlnet-union-sdxl-1.0",
            filename="config_promax.json",
        )

        config = ControlNetModel_Union.load_config(config_file)
        controlnet_model = ControlNetModel_Union.from_config(config)
        
        progress(0.3, desc="Downloading ControlNet model")
        model_file = hf_hub_download(
            "xinsir/controlnet-union-sdxl-1.0",
            filename="diffusion_pytorch_model_promax.safetensors",
        )
        
        progress(0.4, desc="Loading ControlNet model")
        state_dict = load_state_dict(model_file)
        model, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(
            controlnet_model, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0"
        )
        model.to(device=DEVICE, dtype=torch.float16)

        progress(0.6, desc="Loading VAE")
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        ).to(DEVICE)

        progress(0.8, desc="Loading main pipeline")
        pipe = StableDiffusionXLFillPipeline.from_pretrained(
            "SG161222/RealVisXL_V5.0_Lightning",
            torch_dtype=torch.float16,
            vae=vae,
            controlnet=model,
            variant="fp16",
        ).to(DEVICE)

        progress(0.9, desc="Setting up scheduler")
        pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

        progress(1.0, desc="Model loading complete")
        return "Model loaded successfully."
    else:
        # If the model is already loaded, return None instead of a message
        return None

def cleanup_tensors():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()



def fill_image(image, width, height, overlap_percentage, resize_percentage, num_inference_steps, prompt_input, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom, num_images, auto_save, paste_back, guidance_scale):
    global gallery_images
    
    if image is None:
        return (None, None), gr.update(), "Error: No input image provided. Please upload an image first."
    
    try:
        init()
        background, mask = prepare_image_and_mask(image, width, height, overlap_percentage, resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom)
    except Exception as e:
        return (None, None), gr.update(), f"Error preparing image: {str(e)}"

    try:
        cnet_image = background.copy()
        cnet_image.paste(0, (0, 0), mask)

        final_prompt = f"{prompt_input} , high quality, 4k"

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(final_prompt, DEVICE, True)
        
        for n in range(num_images):
            yield (background, cnet_image), gr.update(), f"Generating image {n+1} of {num_images}..."
            
            try:
                for intermediate_image in pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    guidance_scale=guidance_scale,
                    image=cnet_image,
                    num_inference_steps=num_inference_steps
                ):
                    yield (background, intermediate_image), gr.update(), f"Generating image {n+1} of {num_images}..."
                    
                if paste_back:
                    final_image = intermediate_image.convert("RGBA")
                    result_image = background.copy()
                    result_image.paste(final_image, (0, 0), mask)
                else:
                    result_image = intermediate_image.convert("RGBA")
                    
                    
                filename = f"outp_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.png"
                gallery_images.appendleft((result_image, filename))
                if auto_save:
                    save_output(result_image, True, filename)

                while len(gallery_images) > MAX_GALLERY_IMAGES:
                    gallery_images.pop()

                yield (background, result_image), gr.update(value=list(gallery_images)), f"Completed image {n+1} of {num_images}"
            except Exception as e:
                yield (background, None), gr.update(), f"Error generating image {n+1}: {str(e)}"
            
            cleanup_tensors()

        yield (background, result_image), gr.update(value=list(gallery_images)), "All images generated successfully!"
    except Exception as e:
        yield (None, None), gr.update(), f"Error during image generation: {str(e)}"
    
    cleanup_tensors()
    
    
def prepare_image_and_mask(image, width, height, overlap_percentage, resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom):
    target_size = (width, height)

    # Calculate the scaling factor to fit the image within the target size
    scale_factor = min(target_size[0] / image.width, target_size[1] / image.height)
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    
    # Resize the source image to fit within target size
    source = image.resize((new_width, new_height), Image.LANCZOS)

    # Calculate new dimensions based on percentage
    resize_factor = resize_percentage / 100
    new_width = int(source.width * resize_factor)
    new_height = int(source.height * resize_factor)

    # Ensure minimum size of 64 pixels
    new_width = max(new_width, 64)
    new_height = max(new_height, 64)

    # Resize the image
    source = source.resize((new_width, new_height), Image.LANCZOS)

    # Calculate the overlap in pixels based on the percentage
    overlap_x = int(new_width * (overlap_percentage / 100))
    overlap_y = int(new_height * (overlap_percentage / 100))

    # Ensure minimum overlap of 1 pixel
    overlap_x = max(overlap_x, 1)
    overlap_y = max(overlap_y, 1)

    # Calculate margins based on alignment
    if alignment == "Middle":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Left":
        margin_x = 0
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Right":
        margin_x = target_size[0] - new_width
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Top":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = 0
    elif alignment == "Bottom":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = target_size[1] - new_height

    # Adjust margins to eliminate gaps
    margin_x = max(0, min(margin_x, target_size[0] - new_width))
    margin_y = max(0, min(margin_y, target_size[1] - new_height))

    # Create a new background image and paste the resized source image
    background_color = (37, 44, 61) 
    background = Image.new('RGB', target_size, background_color)
    background.paste(source, (margin_x, margin_y))

    # Create the mask
    mask = Image.new('L', target_size, 255)
    mask_draw = ImageDraw.Draw(mask)

    # Calculate overlap areas
    white_gaps_patch = 2

    left_overlap = margin_x + overlap_x if overlap_left else margin_x + white_gaps_patch
    right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width - white_gaps_patch
    top_overlap = margin_y + overlap_y if overlap_top else margin_y + white_gaps_patch
    bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height - white_gaps_patch
    
    if alignment == "Left":
        left_overlap = margin_x + overlap_x if overlap_left else margin_x
    elif alignment == "Right":
        right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width
    elif alignment == "Top":
        top_overlap = margin_y + overlap_y if overlap_top else margin_y
    elif alignment == "Bottom":
        bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height


    # Draw the mask
    mask_draw.rectangle([
        (left_overlap, top_overlap),
        (right_overlap, bottom_overlap)
    ], fill=0)

    return background, mask

def update_preview_mask(image, width, height, overlap_percentage, resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom):
    if image is None:
        return gr.update(value=None), f"No image loaded. Please upload an image first."
    
    try:
        background, mask = prepare_image_and_mask(image, width, height, overlap_percentage, resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom)
        
        preview = background.copy().convert('RGBA')
        red_overlay = Image.new('RGBA', background.size, (255, 0, 0, 64))
        red_mask = Image.new('RGBA', background.size, (0, 0, 0, 0))
        red_mask.paste(red_overlay, (0, 0), mask)
        preview = Image.alpha_composite(preview, red_mask)
        
        return gr.update(value=preview, visible=True), "Preview mask updated."
    except Exception as e:
        return gr.update(visible=False), f"Error updating preview: {str(e)}"


def preload_presets(target_ratio, ui_width, ui_height, resize_percentage):
    if target_ratio == "Portrait":
        changed_width, changed_height = 832, 1216
        ratio_info = "Portrait: result = 832x1216"
    elif target_ratio == "Landscape":
        changed_width, changed_height = 1216, 832
        ratio_info = "Landscape: result = 1216x832"
    elif target_ratio == "Square":
        changed_width, changed_height = 1024, 1024
        ratio_info = "Square: result = 1024x1024"
    # elif target_ratio == "4:3":
        # changed_width, changed_height = 1024, 768
        # ratio_info = "4:3 aspect ratio: result image = 1024x768"
    elif target_ratio == "Custom":
        changed_width, changed_height = ui_width, ui_height
        ratio_info = "Custom aspect ratio: Use sliders to set result image dimensions."
    
    open_advanced = (target_ratio == "Custom")
    # resize_info = f"Input image will be resized to {resize_percentage}% "
    
    return changed_width, changed_height, gr.update(open=open_advanced), f"{ratio_info}\n"


def select_the_right_preset(user_width, user_height):
    if user_width == 832 and user_height == 1216:
        return "Portrait"
    elif user_width == 1216 and user_height == 832:
        return "Landscape"
    elif user_width == 1024 and user_height == 1024:
        return "Square"
    else:
        return "Custom"


def save_output(latest_result, auto_save, filename):
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        full_path = os.path.join(OUTPUT_DIR, filename)
        
        if auto_save:
            if isinstance(latest_result, np.ndarray):
                latest_result = Image.fromarray(latest_result.astype('uint8'))
            latest_result.save(full_path)
            print(f"Image auto-saved as: {full_path}")
            return full_path, filename, f"Image auto-saved as: {full_path}"
        else:
            print(f"Auto-save disabled, assigned filename: {filename}")
            return None, filename
    except Exception as e:
        print(f"Error handling image path/save: {e}")
        return None, None


def update_selected_image(evt: gr.SelectData, gallery_state):
    global selected_image_index
    selected_image_index = evt.index
    if evt.index < len(gallery_state):
        _, filename = gallery_state[evt.index]
        return f"Selected image: {filename}"
    return "Invalid selection"
    
    
def save_selected_image(gallery_state):
    global selected_image_index
    if selected_image_index is None or gallery_state is None:
        return "Please select an image first"
    
    try:
        selected_image, filename = gallery_state[selected_image_index]
        
        print(f"Selected image index: {selected_image_index}")
        print(f"Filename: {filename}")
        
        full_path = os.path.join(OUTPUT_DIR, filename)
        
        if os.path.exists(full_path):
            print(f"File already exists: {full_path}")
            return f"Image already saved as: {filename}"
        
        if isinstance(selected_image, Image.Image):
            selected_image.save(full_path)
        elif isinstance(selected_image, np.ndarray):
            Image.fromarray(selected_image.astype('uint8')).save(full_path)
        elif isinstance(selected_image, str) and os.path.isfile(selected_image):
            import shutil
            shutil.copy(selected_image, full_path)
        else:
            return f"Unsupported image type: {type(selected_image)}"
        
        print(f"Image saved as: {filename}")
        return f"Image saved as: {filename}"
    except Exception as e:
        print(f"Error details: {str(e)}")
        return f"Error saving image: {str(e)}"
 
 
# sends current output image to input  
def use_output_as_input(output_image):
    if output_image is None or output_image[1] is None:
        return gr.update(), "No output image available", "No image selected"
    
    try:
        # We know output_image is a tuple, and we want the second element
        image = output_image[1]
        return gr.update(value=image), "Output image set as input", "Output image set as input"
    except Exception as e:
        return gr.update(), f"Error setting input: {str(e)}", f"Error: {str(e)}"
    
# sends selected gallery image to input  
def send_selected_to_input(gallery_state):
    global selected_image_index
    if selected_image_index is None or gallery_state is None:
        return gr.update(), "Please select an image first", "No image selected"
    
    try:
        selected_image, filename = gallery_state[selected_image_index]
        return gr.update(value=selected_image), f"Image {filename} set as input", f"Selected image: {filename}"
    except Exception as e:
        return gr.update(), f"Error setting input: {str(e)}", f"Error: {str(e)}"
        
  
def open_outputs_folder():
    try:
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        folder_uri = Path(OUTPUT_DIR).absolute().as_uri()
        webbrowser.open(folder_uri)
        return "Opened outputs folder (folder can be shy and hide behind active windows)."
    except Exception as e:
        return f"Error opening outputs folder: {str(e)}"
        

def clear_gallery():
    global gallery_images
    gallery_images.clear()
    return gr.update(value=None), f"Gallery images cleared."


def clear_input_and_result():
    return gr.update(value=None), gr.update(value=None), "Input and result cleared."


def unload_all(progress=gr.Progress()):
    global pipe, vae, controlnet_model
    
    progress(0.1, desc="Starting complete unload process")
    try:
        # Unload pipeline
        if pipe is not None:
            for component in ['unet', 'vae', 'controlnet', 'text_encoder', 'text_encoder_2']:
                if hasattr(pipe, component):
                    delattr(pipe, component)
            del pipe
            pipe = None

        # Unload standalone components
        if 'vae' in globals() and vae is not None:
            del vae
            vae = None

        if 'controlnet_model' in globals() and controlnet_model is not None:
            del controlnet_model
            controlnet_model = None

        # Clear any remaining CUDA cache
        cleanup_tensors()

        progress(1.0, desc="Unload complete")
        return "Unloaded models from VRAM. These will be reloaded as required."
    except Exception as e:
        progress(1.0, desc="Unload failed")
        return f"Error during unload process: {str(e)}"   
     
     
def handle_image_upload(image):
    if image is None:
        return (
            "No image uploaded.", 
            gr.update(interactive=False), 
            gr.update(visible=False), 
            # gr.update(), 
            # gr.update(), 
            gr.update(value=None, placeholder="No image uploaded")
        )
    
    width, height = image.size
    
    info_text = f"""
Original image size: {width}x{height}

Select a Preset or click Custom and set your own.
To help visualize settings - check out Preview Mask in Masking Options below.
"""
    # Reset preview image and update placeholder
    preview_update = gr.update(value=None, placeholder="Click Update Preview Mask")
    
    return (
        info_text, 
        gr.update(interactive=True), 
        gr.update(visible=False), 
        preview_update
    )
 
 
def update_resize_controls(target_ratio, width, height, resize_percentage):
    is_custom = (target_ratio == "Custom")
    width, height, advanced_visible, info = preload_presets(target_ratio, width, height, resize_percentage)
    return (
        gr.update(visible=is_custom),  # manual_resize_options
        gr.update(value=width),  # width_slider
        gr.update(value=height),  # height_slider
        info  # console_info
    )

    
title = """<style>.banner-outpaint{background:linear-gradient(to bottom,#162828,#101c1c);color:#fff;padding:0.75rem;border-radius:0.5rem;border:1px solid rgba(255,255,255,0.1);box-shadow:0 4px 6px rgba(0,0,0,0.1);margin-bottom:0.75rem;text-align:center}.banner-outpaint h1{font-size:1.75rem;margin:0 0 0.25rem 0;font-weight:300;color:#ff6b35 !important}.banner-outpaint p{color:#b0c4c4;font-size:1rem;margin:0 0 0.75rem 0}.banner-outpaint .footer{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;font-size:0.875rem;color:#a0a0a0}.banner-outpaint .powered-by{display:flex;align-items:center;gap:0.25rem}.banner-outpaint .credits{display:flex;flex-direction:column;align-items:flex-end;gap:0.25rem}.banner-outpaint a{color:#4a9eff;text-decoration:none;transition:color 0.2s ease}.banner-outpaint a:hover{color:#6db3ff}@media (max-width:768px){.banner-outpaint .footer{flex-direction:column;gap:0.5rem;align-items:center}.banner-outpaint .credits{align-items:center}}</style><div class="banner-outpaint"><h1>Diffusers Image Outpaint</h1><p>Extend your images using AI</p><div class="footer"><div class="powered-by"><span>⚡ Powered by</span><a href="https://pinokio.computer/" target="_blank">Pinokio</a></div><div class="credits"><div>Code and inspiration borrowed from</div><div><a href="https://huggingface.co/OzzyGT" target="_blank">OzzyGT</a> & <a href="https://huggingface.co/spaces/fffiloni/diffusers-image-outpaint" target="_blank">fffiloni</a></div></div></div></div>"""
    
    
with gr.Blocks() as demo:
    gr.HTML(title)
    with gr.Row():
        input_image = gr.Image(type="pil", label="Input Image")
        result = ImageSlider(interactive=False, label="Generated Image")

    with gr.Row():
        prompt_input = gr.Textbox(label="Prompt (Optional)", scale=3)
        num_images = gr.Slider(value=1, label="Number of Images", minimum=1, maximum=10, step=1, scale=1)
        auto_save = gr.Checkbox(label="Auto-save", value=True, scale=1)
        paste_back = gr.Checkbox(True, label="Paste back original image", scale=1)
        run_button = gr.Button("Generate", variant="primary", scale=2)
        
        with gr.Column(scale=1):
            with gr.Row():
                use_as_input_btn = gr.Button("Use as Input Image", size="sm")
                clear_input_button = gr.Button("Clear", size="sm")
                # unload_all_btn = gr.Button("Unload all Models", variant="stop", size = "sm")



    with gr.Row():
        with gr.Column(scale=1):
            target_ratio = gr.Radio(
                label="Select a Preset",
                choices=["Landscape", "Portrait", "Square", "Custom"],
                value="Landscape",
                scale=1
            )
            resize_percentage = gr.Slider(
                label="Reduce Input Image (relative to output)",
                minimum=20,
                maximum=100,
                step=10,
                value=100,
                interactive=True,
                scale=1
            )

        with gr.Column(scale=1, visible=False) as manual_resize_options:
            width_slider = gr.Slider(
                label="Result Width",
                minimum=512,
                maximum=2048,
                step=8,
                value=1280,
            )

            height_slider = gr.Slider(
                label="Result Height",
                minimum=512,
                maximum=2048,
                step=8,
                value=720,
            )
                    
        with gr.Column(scale=1):
            with gr.Row():  
                num_inference_steps = gr.Slider(label="Steps", minimum=4, maximum=12, step=1, value=8)
                guidance_scale = gr.Slider(value=1.5, label="Guidance Scale", minimum=1.5, maximum=8, step=0.5)
            alignment_dropdown = gr.Dropdown(
                choices=["Middle", "Left", "Right", "Top", "Bottom"],
                value="Middle",
                label="Alignment",
            )

        with gr.Column(scale=1):
            console_info = gr.Textbox(label="Console", lines=4, max_lines=10)
            unload_all_btn = gr.Button("Unload all Models", variant="stop", size = "sm", visible = False)
            
            
    with gr.Accordion("Masking Options - Click to Open or Close", open=False):
        with gr.Row():   
            with gr.Column(scale=1):
                overlap_percentage = gr.Slider(
                    label="Mask overlap (%)",
                    minimum=1,
                    maximum=50,
                    value=5,
                    step=1
                )

                overlap_top = gr.Checkbox(label="Overlap Top", value=True)
                overlap_right = gr.Checkbox(label="Overlap Right", value=True)
                overlap_left = gr.Checkbox(label="Overlap Left", value=True)
                overlap_bottom = gr.Checkbox(label="Overlap Bottom", value=True)
            with gr.Column(scale=1):
                update_preview_btn = gr.Button("Update Preview Mask")
                preview_image = gr.Image(label="Preview Mask", show_download_button=False)

    gallery = gr.Gallery(
        label=f"Image Gallery (most recent {MAX_GALLERY_IMAGES} images)",
        show_label=True,
        elem_id="gallery",
        columns=5,
        height="auto",
        object_fit="contain",
        allow_preview=True,
        preview=False,
        show_download_button=False,
    )

    with gr.Row():
        clear_gallery_btn = gr.Button("Clear Gallery", size="sm", variant="stop", scale=1)
        open_folder_button = gr.Button("Open Outputs Folder", scale=2) 
        gallery_status = gr.Textbox(label="Gallery Status", scale=2, interactive=False)
        save_selected_btn = gr.Button("Save Selected", scale=2)
        send_selected_to_input_btn = gr.Button("Send Selected to Input", scale=1)


    # event handlers  

    run_button.click(
        fn=lambda: "Preparing Image Generation...", 
        inputs=None,
        outputs=console_info,
    ).then(
        fn=init,
        inputs=None,
        outputs=console_info
    ).then(
        fn=fill_image,
        inputs=[
            input_image, width_slider, height_slider, overlap_percentage, 
            resize_percentage, num_inference_steps, prompt_input, 
            alignment_dropdown, overlap_left, overlap_right, overlap_top, 
            overlap_bottom, num_images, auto_save, paste_back, guidance_scale,
        ],
        outputs=[result, gallery, console_info]
    )

    target_ratio.change(
        fn=update_resize_controls,
        inputs=[target_ratio, width_slider, height_slider, resize_percentage],
        outputs=[manual_resize_options, width_slider, height_slider, console_info],
        queue=False
    )

    resize_percentage.change(
        fn=lambda ratio, w, h, r: preload_presets(ratio, w, h, r)[3],  # Only update the info
        inputs=[target_ratio, width_slider, height_slider, resize_percentage],
        outputs=[console_info],
        queue=False
    )
    
    width_slider.change(
        #fn=select_the_right_preset,
        inputs=[width_slider, height_slider],
        outputs=target_ratio,
        queue=False
    )

    height_slider.change(
        #fn=select_the_right_preset,
        inputs=[width_slider, height_slider],
        outputs=target_ratio,
        queue=False
    )

    save_selected_btn.click(
        fn=save_selected_image,
        inputs=gallery,
        outputs=gallery_status
    )  
    
    clear_input_button.click(
        fn=clear_input_and_result,
        inputs=None,
        outputs=[input_image, result, console_info]
    )
    
    gallery.select(
        fn=update_selected_image,
        inputs=gallery,
        outputs=gallery_status
    )  
    
    clear_gallery_btn.click(
        fn=clear_gallery,
        outputs=[gallery, gallery_status]
    )
    
    open_folder_button.click(
        fn=open_outputs_folder,
        inputs=None,
        outputs=gallery_status
    )
    
    unload_all_btn.click(
        fn=unload_all,
        outputs=console_info
    )

    input_image.upload(
        fn=handle_image_upload,
        inputs=[input_image],
        outputs=[console_info, target_ratio, manual_resize_options, preview_image]
    )
    
    update_preview_btn.click(
        fn=update_preview_mask,
        inputs=[input_image, width_slider, height_slider, overlap_percentage, resize_percentage, alignment_dropdown, overlap_left, overlap_right, overlap_top, overlap_bottom],
        outputs=[preview_image, console_info]
    )

    use_as_input_btn.click(
        fn=use_output_as_input,
        inputs=result,
        outputs=[input_image, console_info]
    )
    
    send_selected_to_input_btn.click(
        fn=send_selected_to_input,
        inputs=gallery,
        outputs=[input_image, console_info, gallery_status]
    )
    
    
demo.launch(share=False)