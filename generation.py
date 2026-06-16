"""
generation.py 

This module handles the AI Image Generation phase.
It uses Stable Diffusion (Img2Img) to take a retrieved car image (as a structural blueprint)
and paints over it using the user's text prompt to create a photorealistic, customized car design.
"""

import os
import io
import base64
import numpy as np
import torch
from PIL import Image

# Force matplotlib to run in the background without opening windows
import matplotlib
matplotlib.use('Agg')

# Suppress Hugging Face warnings
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Ensure the output directory exists
GENERATED_DIR = os.path.join("outputs", "generated")
os.makedirs(GENERATED_DIR, exist_ok=True)

# Global variable to store the AI model in memory so we don't have to reload it every time
_pipe = None

def _get_pipe():
    """
    Loads the Stable Diffusion AI model into the computer's memory (RAM or GPU VRAM).
    This only happens once. On subsequent generations, it instantly reuses the loaded model.
    """
    global _pipe
    if _pipe is None:
        try:
            from diffusers import StableDiffusionImg2ImgPipeline
            import torch
            
            # Using Stable Diffusion 1.5 because it is fast and reliable
            model_id = "runwayml/stable-diffusion-v1-5"
            
            # Automatically use the NVIDIA GPU if available, otherwise use the CPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading Stable Diffusion on {device}... (May take time to download if first run)")
            
            # Load the model with optimizations for speed and memory
            _pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None
            )
            _pipe = _pipe.to(device)
            
            if device == "cuda":
                _pipe.enable_attention_slicing()
                
        except Exception as e:
            print(f"Failed to load Stable Diffusion: {e}")
            _pipe = "FALLBACK"
            
    return None if _pipe == "FALLBACK" else _pipe


def generate_image(prompt: str,
                   seed: int = 42,
                   guidance_scale: float = 7.5,
                   num_steps: int = 20,
                   ref_images: list = None) -> Image.Image:
    """
    Generates a new car design.
    Takes the user's creative prompt and the top retrieved image, and merges them using Stable Diffusion.
    """
    pipe = _get_pipe()

    if pipe is not None and ref_images:
        try:
            import torch
            
            # Construct a highly-detailed prompt to ensure professional, photorealistic results
            full_prompt = f"{prompt}. Professional car photography, 8k resolution, highly detailed, photorealistic, cinematic lighting."
            neg_prompt = "blurry, low quality, distorted, cartoon, watermark, text, signature, bad proportions"
            
            generator = torch.Generator(pipe.device).manual_seed(seed)
            
            # Load the top retrieved image to use as the structural baseline
            init_img = Image.open(ref_images[0]).convert("RGB")
            init_img = init_img.resize((512, 512))

            # Run the Stable Diffusion generation process
            result = pipe(
                prompt=full_prompt,
                negative_prompt=neg_prompt,
                image=init_img,
                strength=0.75, # 0.75 means it changes 75% of the original image to match the text prompt
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                generator=generator
            )
            img = result.images[0]
            
        except Exception as e:
            print(f"Generation failed during inference: {e}. Falling back to basic pixel blending.")
            img = _pixel_blend_fallback(ref_images, seed, prompt)
    else:
        img = _pixel_blend_fallback(ref_images, seed, prompt)

    return img


def _pixel_blend_fallback(ref_images: list, seed: int, prompt: str = "") -> Image.Image:
    """
    A basic fallback method that applies standard image processing (color tinting, contrast)
    if the Stable Diffusion AI model fails to load or the computer cannot run it.
    """
    from PIL import ImageEnhance, ImageFilter

    OUT_SIZE = (512, 512)

    base = None
    if ref_images:
        for path in ref_images:
            try:
                img = Image.open(path).convert("RGB")
                w, h = img.size
                side = min(w, h)
                left = (w - side) // 2
                top  = (h - side) // 2
                base = img.crop((left, top, left + side, top + side))
                base = base.resize(OUT_SIZE, Image.LANCZOS)
                break
            except Exception:
                continue

    if base is None:
        return Image.fromarray(np.full((*OUT_SIZE, 3), 80, dtype=np.uint8))

    # Apply color grading based on the prompt's keywords
    arr = np.array(base, dtype=np.float32)
    p = prompt.lower()
    
    if "red" in p:
        arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.4 + 40, 0, 255)
        arr[:, :, 1] = np.clip(arr[:, :, 1] * 0.7, 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] * 0.7, 0, 255)
    elif "green" in p:
        arr[:, :, 0] = np.clip(arr[:, :, 0] * 0.7, 0, 255)
        arr[:, :, 1] = np.clip(arr[:, :, 1] * 1.4 + 40, 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] * 0.7, 0, 255)
    elif "blue" in p:
        arr[:, :, 0] = np.clip(arr[:, :, 0] * 0.7, 0, 255)
        arr[:, :, 1] = np.clip(arr[:, :, 1] * 0.7, 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] * 1.4 + 40, 0, 255)
    else:
        # Default cinematic look
        arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.1 + 10, 0, 255)
        arr[:, :, 1] = np.clip(arr[:, :, 1] * 0.95,     0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] * 1.2 + 20, 0, 255)
        
    base = Image.fromarray(arr.astype(np.uint8))

    # Boost contrast and saturation for a premium look
    base = ImageEnhance.Contrast(base).enhance(1.4)
    base = ImageEnhance.Color(base).enhance(1.3)
    base = ImageEnhance.Sharpness(base).enhance(1.8)
    
    return base


def save_generated(img: Image.Image, name: str) -> str:
    """Saves the generated image to the hard drive."""
    path = os.path.join(GENERATED_DIR, name)
    img.save(path)
    return path


def image_to_base64(img: Image.Image) -> str:
    """Converts the image into a Base64 string so it can be sent directly to the website browser."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
