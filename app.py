import gradio as gr
import torch
import devicetorch
import os
import webbrowser # for opening the outputs folder. seemed the best way for cross-platform compatibility.
import gc
import numpy as np

from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from controlnet_union import ControlNetModel_Union
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline
# from diffusers.utils import is_xformers_available

from huggingface_hub import hf_hub_download
from gradio_imageslider import ImageSlider
from collections import deque
from datetime import datetime
from pathlib import Path
from PIL import Image

DEVICE = devicetorch.get(torch)

MODELS = {
        "RealVisXL V5.0 Lightning": {
        "path": "SG161222/RealVisXL_V5.0_Lightning",
        "default_steps": 6,
        "max_steps": 16,
        "default_guidance": 1.5,
        "max_guidance": 8.0,
        "description": "For fast image generation.",
        "web_link": "https://civitai.com/models/139562/realvisxl-v50"
    },
        "RealismEngineSDXL v30": {
        "path": "misri/realismEngineSDXL_v30VAE",
        "default_steps": 30,
        "max_steps": 50,
        "default_guidance": 4,
        "max_guidance": 10,
        "description": "Realism model.",
        "web_link": "https://civitai.com/models/152525/realism-engine-sdxl"
    },

    # Add other models here following the exact format as above. Only diffusers models, ie from huggingface not civitai.
    # i tried a bunch, but realvisXL V5.0 Lightning seems to work best. good enough until flux and SD3.5 get settled :)
}

# You can change this to any model in your MODELS dictionary
DEFAULT_MODEL = "RealVisXL V5.0 Lightning"  

# If the default model is not in MODELS, use the first model in the dictionary
if DEFAULT_MODEL not in MODELS:
    DEFAULT_MODEL = next(iter(MODELS))
    

MAX_GALLERY_IMAGES = 20 
MIN_IMAGE_SIZE = 512  
OUTPUT_DIR = "outputs" # replace with "C:\path\to\savefolder" if desired.
VAE_SCALE_FACTOR = 8

global pipe
global current_model

# blame Claude!
pipe = None
current_model = None
global_image = None
global_original_image = None 
latest_result = None
gallery_images = deque(maxlen=MAX_GALLERY_IMAGES)
selected_gallery_image = None
selected_image_index = None



def init(model_selection, progress=gr.Progress()):
    global pipe, current_model

    try:
        if pipe is not None and current_model != model_selection:
            progress(0.1, desc="Unloading previous model")
            unload_message = unload_all(progress)
            progress(0.2, desc=unload_message)

        if pipe is None:
            progress(0.3, desc="Starting model initialization")
            
            try:
                progress(0.4, desc="Loading ControlNet configuration")
                config_file = hf_hub_download(
                    "xinsir/controlnet-union-sdxl-1.0",
                    filename="config_promax.json",
                )
                config = ControlNetModel_Union.load_config(config_file)
                controlnet_model = ControlNetModel_Union.from_config(config)
                
                progress(0.5, desc="Downloading ControlNet model")
                model_file = hf_hub_download(
                    "xinsir/controlnet-union-sdxl-1.0",
                    filename="diffusion_pytorch_model_promax.safetensors",
                )
                
                progress(0.6, desc="Loading ControlNet model")
                state_dict = load_state_dict(model_file)
                model, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(
                    controlnet_model, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0"
                )
                model.to(DEVICE, dtype=torch.float16)

                progress(0.7, desc="Loading VAE")
                vae = AutoencoderKL.from_pretrained(
                    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
                ).to(DEVICE)

                progress(0.8, desc=f"Loading main pipeline: {model_selection}.   1st load slow - downloading model and/or loading from storage.")
                pipe = StableDiffusionXLFillPipeline.from_pretrained(
                    MODELS[model_selection]["path"],
                    torch_dtype=torch.float16,
                    controlnet=model,
                    vae=vae,
                    use_safetensors=True,
                    #variant="fp16",  deprecated. and now terminates if model isn't explicitly listed as fp16
                ).to(DEVICE)
                
                progress(0.9, desc="Setting up scheduler")
                pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)  # todo maybe
                
                # if is_xformers_available():
                    # pipe.enable_xformers_memory_efficient_attention()
                
                current_model = model_selection
                progress(1.0, desc="Model loading complete")
                return f"Model {model_selection} loaded successfully."
            except Exception as e:
                return f"Error loading model {model_selection}: {str(e)}"
        else:
            return f"Model {model_selection} is already loaded."
    except Exception as e:
        return f"Unexpected error initializing model: {str(e)}"

    
   
def fill_image(prompt, image, model_selection, guidance_scale, steps, paste_back, auto_save, num_images):
    global latest_result, gallery_images, global_image, pipe
    
    try:
        if image is None:
            return (None, None), gr.update(), "Error: No input image provided. Please upload an image first."
            
        if pipe is None or current_model != model_selection:
            init_message = init(model_selection)
            yield (None, None), gr.update(), f"{init_message} Preparing for image generation..."
        
        source = global_image 
        mask = image["layers"][0]

        source = resize_image(source, min_size=MIN_IMAGE_SIZE, scale_factor=VAE_SCALE_FACTOR)
        mask = mask.resize(source.size, Image.LANCZOS)
        
        alpha_channel = mask.split()[3]
        binary_mask = alpha_channel.point(lambda p: p > 0 and 255)
        cnet_image = source.copy()
        cnet_image.paste(0, (0, 0), binary_mask)

        try:
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(prompt, DEVICE, True)
        except Exception as e:
            yield (None, None), gr.update(), f"Error encoding prompt: {str(e)}"
            return

        all_results = []
        
        for n in range(num_images):
            intermediate_images = []
            
            yield (source, cnet_image), gr.update(), f"Starting generation of image {n+1} of {num_images}..."
            
            try:
                for image in pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    guidance_scale=guidance_scale,
                    num_inference_steps=steps,
                    image=cnet_image,
                ):
                    yield (image, cnet_image), gr.update(), f"Generating image {n+1} of {num_images}..."
                    intermediate_images.append(image)

                final_image = intermediate_images[-1]
                if paste_back:
                    final_image = final_image.convert("RGBA")
                    result_image = cnet_image.copy()
                    result_image.paste(final_image, (0, 0), binary_mask)
                else:
                    result_image = final_image.convert("RGBA")
                    
                all_results.append(result_image)
                gallery_update = update_gallery(result_image, auto_save)
                
                yield (source, result_image), gallery_update, f"Completed image {n+1} of {num_images}"
            
            except Exception as e:
                yield (None, None), gr.update(), f"Error generating image {n+1}: {str(e)}"
            
            finally:
                cleanup_tensors()
            
        if all_results:
            latest_result = all_results[-1]
            yield (source, latest_result), gallery_update, "All images generated successfully!"
        else:
            yield (source, source), gallery_update, "No images were generated successfully."
        
    except Exception as e:
        yield (None, None), gr.update(), f"Unexpected error during image generation: {str(e)}"
    
    finally:
        cleanup_tensors()
    
# Clear CUDA cache
def cleanup_tensors():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
    

# via UI button to release resources while between tasks. called to clear out previous pipe when changing model
def unload_all(progress=gr.Progress()):
    global pipe, current_model
    
    progress(0.1, desc="Starting unload process")

    try:
        # Unload pipeline
        if pipe is not None:
            for component in ['unet', 'vae', 'controlnet', 'text_encoder', 'text_encoder_2', 'scheduler']:
                if hasattr(pipe, component):
                    setattr(pipe, component, None)
            del pipe
            pipe = None

        # Reset current model
        current_model = None

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # Force garbage collection
        gc.collect()

        progress(1.0, desc="Unload complete")
        return "Models unloaded and memory cleared. New models will be loaded when needed."
    except Exception as e:
        progress(1.0, desc="Unload failed")
        return f"Error during unload process: {str(e)}"

   
# from UI button    
def send_to_input(result_slider):
    if latest_result is not None and result_slider is not None:
        global global_image
        global_image = latest_result
        resize_slider, resize_button, info = update_resize_controls({"background": latest_result})
        return gr.update(value=latest_result), gr.update(value=None), resize_slider, resize_button, info

    return gr.update(), gr.update(), gr.update(), gr.update(), "No result to send to input"


# from UI button  
def clear_input_and_result():
    return gr.update(value=None), gr.update(value=None), "Input and result cleared." 
    
    

# Sizing Functions. Yes, requires refactoring. may add upscaling options and presets

# triggered on new input image for sizing info    
def handle_image_upload(image):
    if image is None:
        return gr.update(value=None, choices=[], interactive=False), gr.update(interactive=False), "No image loaded. Please upload an image."
    
    # Process the image
    resize_slider, resize_button, info = update_resize_controls(image)
    return resize_slider, resize_button, info

    
# Resize slider.  checks for available resize options and displays relavent info to console
def preview_resize(percentage):
    global global_image, global_original_image
    if global_original_image is None:
        return "No image loaded. Please upload an image first."
    
    if percentage == 100:
        _, _, info_text = update_resize_controls({"background": global_original_image})
        return info_text
    
    original_w, original_h = global_original_image.size
    current_w, current_h = global_image.size
    new_w = ((original_w * percentage) // 100 // VAE_SCALE_FACTOR) * VAE_SCALE_FACTOR
    new_h = ((original_h * percentage) // 100 // VAE_SCALE_FACTOR) * VAE_SCALE_FACTOR
    
    if new_w < MIN_IMAGE_SIZE or new_h < MIN_IMAGE_SIZE:
        return f"Cannot resize below minimum allowed size of {MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE}."
    
    mp = (new_w * new_h) / 1000000
    estimated_vram = 10.0 + (mp * 4) + 1.0
    
    if percentage == 100:
        resize_info = '<p style="color: #4CAF50;">Image is at 100% of its original size. No resize needed.</p>'
    elif new_w == current_w and new_h == current_h:
        resize_info = '<p style="color: #FF6347;">Preview size = current size. No resize required.</p>'
    else:
        resize_info = '<p style="color: #4CAF50;">Click \'Apply Resize\' to perform this resize.</p>'
    
    return f"""
<p><span style="color: #FF8C00; font-weight: bold;">Preview for {percentage}% of original size:</span></p>
<p>New size would be: <span style="color: #3498DB;">{new_w}x{new_h}</span> (current size: <span style="color: #3498DB;">{current_w}x{current_h}</span>)</p>
<p>Estimated VRAM usage: <span style="color: #9932CC;">{estimated_vram:.1f}GB</span></p>
{resize_info}
"""

# pre-calculating resize for update_resize_controls
def calculate_resize_options(image):
    if image is None:
        return []
    
    original_w, original_h = image.size
    aspect_ratio = original_w / original_h
    
    # Define a range of percentages to check
    percentages = list(range(100, 9, -10))  # 100%, 90%, 80%, ..., 10%
    
    valid_options = []
    min_dimension = min(original_w, original_h)
    
    for percentage in percentages:
        new_w = int((original_w * percentage) / 100)
        new_h = int((original_h * percentage) / 100)
        
        # Ensure dimensions are multiples of VAE_SCALE_FACTOR
        new_w = (new_w // VAE_SCALE_FACTOR) * VAE_SCALE_FACTOR
        new_h = (new_h // VAE_SCALE_FACTOR) * VAE_SCALE_FACTOR
        
        # Check if resized dimensions meet minimum requirements
        if new_w >= MIN_IMAGE_SIZE and new_h >= MIN_IMAGE_SIZE:
            valid_options.append(percentage)
        else:
            # If we've hit the minimum size, calculate the exact percentage needed
            if not valid_options:
                min_percentage = max((MIN_IMAGE_SIZE / original_w) * 100, (MIN_IMAGE_SIZE / original_h) * 100)
                min_percentage = math.ceil(min_percentage)
                valid_options.append(min_percentage)
            break
    
    # Remove 100% if it's the only option
    if len(valid_options) == 1 and valid_options[0] == 100:
        return []
    
    # Ensure we have at least 3 options (if possible) by adding intermediate values
    while len(valid_options) < 3 and len(valid_options) < len(percentages):
        for i in range(len(valid_options) - 1):
            mid_percentage = (valid_options[i] + valid_options[i+1]) // 2
            if mid_percentage not in valid_options:
                valid_options.insert(i+1, mid_percentage)
                if len(valid_options) == 3:
                    break
    
    return sorted(valid_options, reverse=True)

# updates the sizer and console with the available options for the newly loaded image
def update_resize_controls(image):
    global global_image, global_original_image
    if image is None or (isinstance(image, dict) and image.get("background") is None):
        return (
            gr.update(value=None, choices=[], interactive=False),
            gr.update(interactive=False),
            "No image loaded. Please upload an image."
        )
    
    if isinstance(image, dict):
        image = image["background"]
    
    global_original_image = image.copy()
    global_image = global_original_image.copy()
    
    width, height = global_original_image.size
    mp = (width * height) / 1000000
    estimated_vram = 10.0 + (mp * 4) + 1.0
    
    resize_options = calculate_resize_options(global_original_image)
    
    # Always include 100% in the choices
    if 100 not in resize_options:
        resize_options = [100] + resize_options
    
    info_text = f"""
<p>Current size: <span style="color: #FF8C00; font-weight: bold;">{width}x{height}</span></p>
<p>Available resize options: <span style="color: #4CAF50;">{", ".join(f"{opt}%" for opt in resize_options)}</span></p>
<p>Estimated VRAM usage: <span style="color: #FF3333;">{estimated_vram:.1f}GB</span></p>
<p>Minimum allowed size: <span style="color: #4CAF50;">{MIN_IMAGE_SIZE} pixels</span> for the smallest dimension</p>
"""
    
    return (
        gr.update(value=100, choices=resize_options, interactive=True),
        gr.update(interactive=bool(resize_options)),
        info_text
    )
 
# from resize button. always applying resize to original image NEVER to a copy
def apply_resize(percentage):
    global global_image, global_original_image
    if global_original_image is None:
        return None, "No image loaded", gr.update(), gr.update()
    
    original_w, original_h = global_original_image.size
    new_w = ((original_w * percentage) // 100 // VAE_SCALE_FACTOR) * VAE_SCALE_FACTOR
    new_h = ((original_h * percentage) // 100 // VAE_SCALE_FACTOR) * VAE_SCALE_FACTOR
    
    if percentage == 100:
        global_image = global_original_image.copy()
        message = f"Image restored to original copy: <span style=\"color: #3498DB;\">{original_w}x{original_h}</span>"
    else:
        global_image = global_original_image.resize((new_w, new_h), Image.LANCZOS)
        message = f"Image resized to <span style=\"color: #3498DB;\">{new_w}x{new_h}</span>"

    resize_slider, resize_button, info = update_resize_controls({"background": global_image})
    
    return global_image, message + "<br>" + info, resize_slider, resize_button


# from fill_image -redundant?
def resize_image(image, min_size=MIN_IMAGE_SIZE, scale_factor=VAE_SCALE_FACTOR):
    width, height = image.size
    
    # Ensure minimum size
    if width < min_size or height < min_size:
        scale = max(min_size / width, min_size / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
    else:
        new_width, new_height = width, height
    
    # Round to nearest multiple of scale_factor
    new_width = ((new_width + scale_factor - 1) // scale_factor) * scale_factor
    new_height = ((new_height + scale_factor - 1) // scale_factor) * scale_factor
    
    return image.resize((new_width, new_height), Image.LANCZOS)
    
    
    
# Gallery functions:

# from fill_image
def update_gallery(result_image, auto_save):
    global gallery_images
    
    filename = f"inp_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.png"
    gallery_images.appendleft((result_image, filename))
    
    if auto_save:
        save_output(result_image, True, filename)
    
    while len(gallery_images) > MAX_GALLERY_IMAGES:
        gallery_images.pop()
    
    return gr.update(value=list(gallery_images))
    
# from UI button    
def clear_gallery():
    global gallery_images, selected_gallery_image
    gallery_images.clear()
    selected_gallery_image = None
    return gr.update(value=None), gr.update(value=None), "Selected image no longer in gallery"
    
# from UI button   
def open_outputs_folder():
    try:
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        folder_uri = Path(OUTPUT_DIR).absolute().as_uri()
        webbrowser.open(folder_uri)
        return "Opened outputs folder (folder can be shy and hide behind active windows)."
    except Exception as e:
        return f"Error opening outputs folder: {str(e)}"
        
# from gallery.select        
def update_selected_image(evt: gr.SelectData):
    global selected_gallery_image, selected_image_index
    if evt.index < len(gallery_images):
        selected_image, filename = gallery_images[evt.index]
        selected_gallery_image = selected_image
        selected_image_index = evt.index
        return f"Selected image: {filename}"
    return "Invalid selection"

# from UI button 
def save_selected_image(gallery_state):
    global selected_image_index
    if selected_image_index is None or gallery_state is None:
        return "Please select an image first"
    
    try:
        selected_image, filename = gallery_state[selected_image_index]
        full_path = os.path.join(OUTPUT_DIR, filename)
        
        if os.path.exists(full_path):
            return f"Image already saved as: {filename}"
        
        if isinstance(selected_image, np.ndarray):
            Image.fromarray(selected_image).save(full_path)
        elif isinstance(selected_image, Image.Image):
            selected_image.save(full_path)
        elif isinstance(selected_image, str):
            if os.path.exists(selected_image):
                import shutil
                shutil.copy(selected_image, full_path)
            else:
                return f"Error: Invalid image data (string) for {filename}"
        else:
            return f"Error: Invalid image data for {filename}. Type: {type(selected_image)}"
        
        return f"Image saved as: {filename}"
    except Exception as e:
        print(f"Error details: {str(e)}")
        return f"Error saving image: {str(e)}"
        
# from UI button 
def send_selected_to_input():
    global selected_gallery_image, global_image, global_original_image
    try:
        if selected_gallery_image is not None:
            global_image = selected_gallery_image.copy()
            global_original_image = selected_gallery_image.copy()
            
            # Update resize controls for the new image
            resize_slider_update, resize_button_update, console_info_update = update_resize_controls({"background": global_image})
            
            return (
                gr.update(value=global_image),  # Update input image
                resize_slider_update,  # Update resize slider options and value
                resize_button_update,  # Update resize button state
                console_info_update,  # Update console info
                "Selected image successfully sent to input"  # Status message
            )
        else:
            return gr.update(), gr.update(value=None, choices=[]), gr.update(), gr.update(), "No image selected. Please select an image from the gallery first."
    except Exception as e:
        print(f"Error sending image to input: {str(e)}")
        return gr.update(), gr.update(value=None, choices=[], interactive=False), gr.update(interactive=False), gr.update(), f"Error sending image to input: {str(e)}"
        

# combine function with save_selected_image?
def save_output(latest_result, auto_save, filename):
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        full_path = os.path.join(OUTPUT_DIR, filename)
        
        if auto_save:
            latest_result.save(full_path)
            print(f"Image auto-saved as: {full_path}")
            return full_path, filename
        else:
            print(f"Auto-save disabled, assigned filename: {filename}")
            return None, filename
    except Exception as e:
        print(f"Error handling image path/save: {e}")
        return None, None
 
  
# update console with model info from library          
def update_model_info(model_selection):
    model_info = MODELS[model_selection]
    sea_green = "#20B2AA"  # Light Sea Green color
    
    # Create the clickable link
    web_link_html = f'<a href="{model_info["web_link"]}" target="_blank">View Model</a>' if model_info["web_link"] else "Not available"
    
    console_content = f"""
    <p><strong style="color: {sea_green}; font-size: 1.1em;">Model:</strong> {model_selection}</p>
    <p><strong style="color: {sea_green}; font-size: 1.1em;">Description:</strong> {model_info['description']}</p>
    <p><strong style="color: {sea_green}; font-size: 1.1em;">Recommended Settings:</strong></p>
    <ul style="margin-top: 5px;">
        <li><strong>Steps:</strong> {model_info['default_steps']} (max: {model_info['max_steps']})</li>
        <li><strong>Guidance Scale:</strong> {model_info['default_guidance']} (max: {model_info['max_guidance']})</li>
    </ul>
    <p><strong style="color: {sea_green}; font-size: 1.1em;">Web Link:</strong> {web_link_html}</p>
    """
    # Wrap in the scrollable div
    console_html = f'<div class="scrollable-console">{console_content}</div>'
    
    return (
        gr.update(value=model_info["default_steps"], maximum=model_info["max_steps"]),
        gr.update(value=model_info["default_guidance"], maximum=model_info["max_guidance"]),
        console_html
    )

# update UI elements when changing models    
def handle_model_change(model_selection):
    steps, guidance_scale, console_info = update_model_info(model_selection)
    # console_info += "<p>Model settings updated. The new model will be loaded when you start generation.</p>"
    return steps, guidance_scale, console_info
  

#UI title bar  
title = """
<style>
.title-container{text-align:center;margin:auto;padding:8px 12px;background:linear-gradient(to bottom,#162828,#101c1c);color:#fff;border-radius:8px;font-family:Arial,sans-serif;border:2px solid #0a1212;box-shadow:0 2px 4px rgba(0,0,0,0.1);position:relative}.title-container h1{font-size:2em;margin:0 0 5px;font-weight:300;color:#ff6b35}.title-container p{color:#b0c4c4;font-size:0.9em;margin:0 0 5px}.title-container a{color:#ff6b35;text-decoration:none;transition:color 0.3s ease}.title-container a:hover{color:#ff8c5a}.links-left,.links-right{position:absolute;bottom:5px;font-size:0.8em;color:#a0a0a0}.links-left{left:10px}.links-right{right:10px}.emoji-icon{vertical-align:middle;margin-right:3px;font-size:1em}
</style>
<div class="title-container">
<h1>Diffusers Image Inpaint/Remove</h1>
<p>Select an image - Draw a mask - Remove or replace.</p>
<div class="links-left"><span class="emoji-icon">⚡</span>Powered by <a href="https://pinokio.computer/" target="_blank">Pinokio</a></div>
<div class="links-right">Core code borrowed from and inspired by <a href="https://huggingface.co/OzzyGT" target="_blank">OzzyGT</a></div>
</div>
"""


#CSS style for scrollable console. added to limit gr.HTML vertical growth
css = """
.scrollable-console {
    max-height: 130px;  /* Adjust this value as needed */
    overflow-y: auto;
    border: 1px solid #ccc;
    padding: 10px;

}
"""

with gr.Blocks(css=css) as demo:
    gr.HTML(title)
    with gr.Row():
        input_image = gr.ImageMask(
            type="pil",
            label="Load an image and draw a mask with the Draw Tool",
            layers=False,
            sources=["upload"],
        )
        result = ImageSlider(
            interactive=False,
            label="Generated Image",
            elem_classes=["image-slider-custom"],
        )
        
    with gr.Row():
        prompt = gr.Textbox(value="high quality, 4K", label="Prompt (add details for inpaint)", scale=2)
        num_images = gr.Slider(value=1, label="Number of images", minimum=1, maximum=10, step=1)
        auto_save = gr.Checkbox(True, label="Auto-save")
        paste_back = gr.Checkbox(True, label="Paste back background")
        run_button = gr.Button("Generate", variant="primary", scale=1)
            
        with gr.Column(scale=1):
            with gr.Row():
                result_to_input = gr.Button("Use as Input", size="sm")
                clear_input_button = gr.Button("Clear", size="sm")
                unload_all_btn = gr.Button("Unload models", variant="stop", size="sm")
     
        
    with gr.Row():
        model_selection = gr.Dropdown(
            choices=list(MODELS.keys()),
            value=DEFAULT_MODEL,
            label="Model"
        )
        with gr.Column(scale=1):
            steps = gr.Slider(
                value=MODELS[DEFAULT_MODEL]["default_steps"],
                label="Steps",
                minimum=1,
                maximum=MODELS[DEFAULT_MODEL]["max_steps"],
                step=1,
                visible=True
            )
            guidance_scale = gr.Slider(
                value=MODELS[DEFAULT_MODEL]["default_guidance"],
                label="Guidance Scale",
                minimum=1.5,
                maximum=MODELS[DEFAULT_MODEL]["max_guidance"],
                step=0.5
            )
        with gr.Column(scale=1):
            console_info = gr.HTML(
                value='<div class="scrollable-console"></div>',
                label="Console"
            )
            
        with gr.Column(scale=1):
            resize_slider = gr.Dropdown(
                choices=[],
                value=None,
                label="Resize Options (%)",
                interactive=False
            )
            resize_button = gr.Button("Apply Resize", size="sm", interactive=False)


        
    with gr.Row():
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
            interactive=False,
        )
        
    with gr.Row():
        clear_gallery_btn = gr.Button("Clear Gallery", size="sm", variant="stop", scale=1)
        open_folder_button = gr.Button("Open Outputs Folder", scale=2) 
        gallery_status = gr.Textbox(interactive=False, scale=2)
        save_selected_btn = gr.Button("Save Selected", scale=2)
        send_selected_to_input_btn = gr.Button("Send Selected to Input", scale=1)  
   

    # event handlers        

    run_button.click(
        fn=lambda model: init(model),
        inputs=[model_selection],
        outputs=console_info
    ).then(
        fn=fill_image,
        inputs=[prompt, input_image, model_selection, guidance_scale, steps, paste_back, auto_save, num_images],
        outputs=[result, gallery, console_info]
    )
    
    clear_gallery_btn.click(
        fn=clear_gallery,
        outputs=[gallery, gallery_status]
    )
    
    clear_input_button.click(
        fn=clear_input_and_result,
        outputs=[input_image, result, console_info]
    )
    
    result_to_input.click(
        fn=send_to_input,
        inputs=[result], 
        outputs=[input_image, result, resize_slider, resize_button, console_info],
    )
    
    open_folder_button.click(
        fn=open_outputs_folder,
        inputs=None,
        outputs=console_info
    )

    input_image.upload(
        fn=handle_image_upload,
        inputs=[input_image],
        outputs=[resize_slider, resize_button, console_info]
    )

    resize_slider.change(
        fn=preview_resize,
        inputs=[resize_slider],
        outputs=[console_info]
    )

    resize_button.click(
        fn=apply_resize,
        inputs=[resize_slider],
        outputs=[input_image, console_info, resize_slider, resize_button]
    )

    unload_all_btn.click(
        fn=unload_all,
        outputs=[console_info]
    )
    
    gallery.select(
        fn=update_selected_image,
        outputs=gallery_status
    )

    save_selected_btn.click(
        fn=save_selected_image,
        inputs=gallery,
        outputs=gallery_status
    )

    send_selected_to_input_btn.click(
        fn=send_selected_to_input,
        inputs=None,
        outputs=[
            input_image,
            resize_slider,
            resize_button,
            console_info,
            gallery_status
        ]
    )
    
    model_selection.change(
        fn=handle_model_change,
        inputs=[model_selection],
        outputs=[steps, guidance_scale, console_info]
    )
    
demo.launch(share=False)
