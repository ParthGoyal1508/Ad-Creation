"""
ad_generator.py
---------------
LLM 2: Uses HuggingFace InferenceClient with Stable Diffusion XL (text-to-image)
to generate advertisement visuals from the enhanced prompt produced by LLM 1.
Supports post-processing: text overlay, resizing, and saving in multiple formats.
"""

import os
from huggingface_hub import InferenceClient
from PIL import Image, ImageDraw, ImageFont

# Model used for image generation via HuggingFace InferenceClient
HF_IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

# Standard output resolution for advertisements (1:1 ratio suitable for most platforms)
OUTPUT_WIDTH = 1024
OUTPUT_HEIGHT = 1024

def generate_advertisement(
    image_prompt: str,
    ad_text: str,
    hf_api_key: str,
    output_dir: str,
    num_images: int = 1,
    template_path: str = None
) -> list:
    """
    Generates advertisement images using Stable Diffusion 2.1 via HuggingFace API.
    Optionally blends with a provided template image and overlays the ad text.

    Args:
        image_prompt  (str):  Enhanced prompt from LLM 1 (prompt_builder).
        ad_text       (str):  Original advertisement text for overlay.
        hf_api_key    (str):  HuggingFace API key.
        output_dir    (str):  Directory path to save generated images.
        num_images    (int):  Number of advertisement variants to generate.
        template_path (str):  Optional path to an existing ad template image.

    Returns:
        list: List of file paths for all saved advertisement images.
    """

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    saved_paths = []

    # InferenceClient handles new HuggingFace routing automatically
    client = InferenceClient(api_key=hf_api_key)

    # Load template image if provided (used for background blending)
    template_image = None
    if template_path and os.path.exists(template_path):
        print(f"[Loader] Loading template image from: {template_path}")
        template_image = Image.open(template_path).convert("RGBA")
        template_image = template_image.resize((OUTPUT_WIDTH, OUTPUT_HEIGHT))

    for i in range(num_images):
        print(f"\n[LLM 2 - Stable Diffusion XL] Generating image variant {i + 1}/{num_images}...")

        try:
            # Call HuggingFace text-to-image via InferenceClient
            generated_image = client.text_to_image(
                prompt=image_prompt,
                model=HF_IMAGE_MODEL,
                width=OUTPUT_WIDTH,
                height=OUTPUT_HEIGHT,
                num_inference_steps=30,
                guidance_scale=7.5
            )

            # Convert PIL Image returned by InferenceClient to RGBA and resize
            generated_image = generated_image.convert("RGBA")
            generated_image = generated_image.resize((OUTPUT_WIDTH, OUTPUT_HEIGHT))

            # If a template was provided, blend it with the generated image
            if template_image is not None:
                generated_image = _blend_with_template(generated_image, template_image)

            # Overlay the advertisement text on the image
            final_image = _overlay_text(generated_image, ad_text)

            # Convert to RGB for saving as JPEG/PNG
            final_rgb = final_image.convert("RGB")

            # Save the final advertisement image
            output_filename = f"advertisement_variant_{i + 1}.png"
            output_path = os.path.join(output_dir, output_filename)
            final_rgb.save(output_path, format="PNG", optimize=True)

            saved_paths.append(output_path)
            print(f"[Saved] Advertisement saved to: {output_path}")

        except Exception as e:
            print(f"[Error] Image generation failed for variant {i + 1}: {e}")
            print(f"[Hint] If error is 503, the model is loading - wait 20 seconds and retry.")

    return saved_paths

def _blend_with_template(generated: Image.Image, template: Image.Image) -> Image.Image:
    """
    Blends the AI-generated image with the provided template using alpha compositing.
    The template acts as a branded frame/overlay on top of the generated image.

    Args:
        generated (PIL.Image): The AI-generated advertisement image.
        template  (PIL.Image): The user-provided template image (RGBA).

    Returns:
        PIL.Image: Blended composite image.
    """
    # Make template semi-transparent (40% opacity) to let generated image show through
    r, g, b, a = template.split()
    a = a.point(lambda p: int(p * 0.4))
    template_transparent = Image.merge("RGBA", (r, g, b, a))

    # Composite: generated image as base, template overlay on top
    blended = Image.alpha_composite(generated, template_transparent)
    return blended

def _overlay_text(image: Image.Image, ad_text: str) -> Image.Image:
    """
    Overlays the advertisement text at the bottom of the image with a
    semi-transparent dark banner for readability.

    Args:
        image   (PIL.Image): The generated advertisement image (RGBA).
        ad_text (str):       The advertisement text to overlay.

    Returns:
        PIL.Image: Image with text overlay applied.
    """
    # Create a drawing context on a copy of the image
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    img_width, img_height = img_copy.size

    # Wrap long text to fit within the image width
    wrapped_text = _wrap_text(ad_text, max_chars=55)
    line_count = len(wrapped_text.split("\n"))

    # Calculate banner height based on number of text lines
    banner_height = (line_count * 40) + 30
    banner_top = img_height - banner_height

    # Draw a semi-transparent dark rectangle as text background
    banner_overlay = Image.new("RGBA", img_copy.size, (0, 0, 0, 0))
    banner_draw = ImageDraw.Draw(banner_overlay)
    banner_draw.rectangle(
        [(0, banner_top), (img_width, img_height)],
        fill=(0, 0, 0, 170)  # Black with ~67% opacity
    )
    img_copy = Image.alpha_composite(img_copy, banner_overlay)
    draw = ImageDraw.Draw(img_copy)

    # Load a font; fall back to default PIL font if custom font unavailable
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=32)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=32)
        except (IOError, OSError):
            font = ImageFont.load_default()

    # Draw the wrapped advertisement text centered horizontally
    text_y = banner_top + 15
    for line in wrapped_text.split("\n"):
        # Calculate horizontal center position for each line
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = (img_width - text_width) // 2
        draw.text((text_x, text_y), line, fill=(255, 255, 255, 255), font=font)
        text_y += 40

    return img_copy

def _wrap_text(text: str, max_chars: int = 55) -> str:
    """
    Wraps text into multiple lines so it fits within the image width.

    Args:
        text      (str): The text to wrap.
        max_chars (int): Maximum characters per line.

    Returns:
        str: Text with newline characters inserted at appropriate positions.
    """
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        # Check if adding the next word exceeds the line limit
        if len(current_line) + len(word) + 1 <= max_chars:
            current_line += (" " + word) if current_line else word
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return "\n".join(lines)
