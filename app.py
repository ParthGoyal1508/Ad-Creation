"""
app.py
------
Gradio UI for Capstone Project CS[03]: Advertisement Creator using Image Generation.
Designed for deployment on Hugging Face Spaces.

Features:
  - Ad text input with style selector
  - Optional template image upload
  - Multi-image generation (1–4 variants)
  - Enhanced prompt preview (LLM 1 output transparency)
  - Image gallery with download support
  - HF API key input (or reads from HF_API_KEY env var)

Run locally:
    pip install -r requirements.txt
    python app.py

Deploy on HF Spaces:
    Push this file + prompt_builder.py + ad_generator.py + requirements.txt
    to a Hugging Face Space (Gradio SDK).
    Set HF_API_KEY as a Space Secret.
"""

import os
import gradio as gr
from PIL import Image

from prompt_builder import build_image_prompt
from ad_generator import generate_advertisement

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_STYLES = [
    "modern",
    "minimalist",
    "vibrant",
    "luxury",
    "retro",
    "corporate",
    "playful",
    "cinematic",
]

OUTPUT_DIR = "./output"

STYLE_DESCRIPTIONS = {
    "modern":     "Clean lines, bold typography, contemporary feel",
    "minimalist": "White space, simplicity, elegant restraint",
    "vibrant":    "Bold colors, high energy, eye-catching contrast",
    "luxury":     "Premium feel, gold tones, sophisticated elegance",
    "retro":      "Vintage aesthetics, nostalgic color palettes",
    "corporate":  "Professional, trustworthy, business-focused",
    "playful":    "Fun, bright, cheerful, appeals to younger audiences",
    "cinematic":  "Dramatic lighting, film-quality, storytelling mood",
}

# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------
def generate_ads(
    ad_text: str,
    style: str,
    num_images: int,
    template_image,
    hf_api_key: str,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Main generation pipeline called by the Gradio interface.

    Args:
        ad_text       : Advertisement description from user
        style         : Visual style choice
        num_images    : Number of variants to generate (1–4)
        template_image: Optional uploaded template image path
        hf_api_key    : HuggingFace API key (falls back to env var)
        progress      : Gradio progress tracker

    Returns:
        Tuple of (image_list, enhanced_prompt_text, status_message)
    """

    # --- Input validation ---
    if not ad_text or not ad_text.strip():
        return [], "", "❌ Please enter advertisement text before generating."

    # Resolve API key: UI input > environment variable
    api_key = (hf_api_key or "").strip() or os.environ.get("HF_API_KEY", "")
    if not api_key:
        return [], "", (
            "❌ HuggingFace API key is required.\n"
            "Enter it in the API Key field, or set the HF_API_KEY environment variable.\n"
            "Get a free key at: https://huggingface.co/settings/tokens"
        )

    num_images = int(num_images)

    try:
        # ----------------------------------------------------------------
        # Step 1 — LLM 1: Enhance ad text into a rich SD prompt
        # ----------------------------------------------------------------
        progress(0.1, desc="Step 1/2 — Engineering prompt with Qwen2.5-7B...")
        enhanced_prompt = build_image_prompt(
            ad_text=ad_text,
            style=style,
            hf_api_key=api_key,
        )

        # ----------------------------------------------------------------
        # Step 2 — LLM 2: Generate advertisement images via SDXL
        # ----------------------------------------------------------------
        progress(0.4, desc="Step 2/2 — Generating images with Stable Diffusion XL...")
        saved_paths = generate_advertisement(
            image_prompt=enhanced_prompt,
            ad_text=ad_text,
            hf_api_key=api_key,
            output_dir=OUTPUT_DIR,
            num_images=num_images,
            template_path=template_image,
        )

        # Load generated images as PIL objects for the gallery
        progress(0.95, desc="Loading generated images...")
        images = []
        for path in saved_paths:
            if os.path.exists(path):
                images.append(Image.open(path))

        if not images:
            return [], enhanced_prompt, (
                "⚠️ No images were generated. "
                "This may be due to an API error or model loading (503). "
                "Please wait 20 seconds and try again."
            )

        progress(1.0, desc="Done!")
        status = f"✅ Successfully generated {len(images)} advertisement variant(s)!"
        return images, enhanced_prompt, status

    except Exception as e:
        error_msg = str(e)
        if "503" in error_msg:
            return [], "", (
                "⚠️ Model is loading (HTTP 503). "
                "HuggingFace free-tier models spin down when idle. "
                "Please wait 20–30 seconds and try again."
            )
        return [], "", f"❌ Generation failed: {error_msg}"


# ---------------------------------------------------------------------------
# Gradio UI Layout
# ---------------------------------------------------------------------------
def build_ui():
    """Constructs and returns the Gradio Blocks interface."""

    css = """
    #title-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 8px;
    }
    #title-banner h1 {
        color: #e2b96f;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 6px 0;
    }
    #title-banner p {
        color: #a0aec0;
        font-size: 0.95rem;
        margin: 0;
    }
    #generate-btn {
        background: linear-gradient(135deg, #e2b96f, #c9963b) !important;
        color: #1a1a2e !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        border: none !important;
        border-radius: 8px !important;
        height: 52px !important;
    }
    #generate-btn:hover {
        opacity: 0.92 !important;
        transform: translateY(-1px) !important;
    }
    #status-box textarea {
        font-size: 0.9rem;
    }
    .model-badge {
        display: inline-block;
        background: #2d3748;
        color: #81e6d9;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        margin: 2px;
    }
    """

    with gr.Blocks(
        css=css,
        title="AI Advertisement Creator",
        theme=gr.themes.Soft(
            primary_hue="amber",
            secondary_hue="blue",
            neutral_hue="slate",
        ),
    ) as demo:

        # -----------------------------------------------------------------
        # Header Banner
        # -----------------------------------------------------------------
        gr.HTML("""
        <div id="title-banner">
            <h1>🎨 AI Advertisement Creator</h1>
            <p>
                Powered by
                <span class="model-badge">Qwen2.5-7B-Instruct</span>
                <span class="model-badge">Stable Diffusion XL</span>
                &nbsp;|&nbsp; Capstone Project CS[03]
            </p>
        </div>
        """)

        # -----------------------------------------------------------------
        # Main Layout — Inputs (left) | Outputs (right)
        # -----------------------------------------------------------------
        with gr.Row(equal_height=False):

            # ---- LEFT COLUMN: Inputs ----
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### ✏️ Advertisement Details")

                ad_text_input = gr.Textbox(
                    label="Advertisement Text",
                    placeholder=(
                        "e.g. Nike Air Max — Engineered for champions.\n"
                        "Describe your product, message, and target audience."
                    ),
                    lines=4,
                    max_lines=8,
                )

                style_input = gr.Dropdown(
                    label="Visual Style",
                    choices=VALID_STYLES,
                    value="modern",
                    info="Choose the visual aesthetic for your advertisement.",
                )

                # Dynamic style description hint
                style_hint = gr.Markdown(
                    value=f"💡 *{STYLE_DESCRIPTIONS['modern']}*",
                    visible=True,
                )

                num_images_input = gr.Slider(
                    label="Number of Variants",
                    minimum=1,
                    maximum=4,
                    step=1,
                    value=1,
                    info="Generate up to 4 advertisement variants in one run.",
                )

                gr.Markdown("### 🖼️ Template Image (Optional)")
                template_input = gr.Image(
                    label="Upload Brand Template",
                    type="filepath",
                    sources=["upload"],
                    height=160,
                )
                gr.Markdown(
                    "<small>Upload an existing ad template to blend with the AI-generated image.</small>",
                    visible=True,
                )

                gr.Markdown("### 🔑 API Configuration")
                api_key_input = gr.Textbox(
                    label="HuggingFace API Key",
                    placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    type="password",
                    info=(
                        "Leave blank if HF_API_KEY is set as an environment variable. "
                        "Get a free key at huggingface.co/settings/tokens"
                    ),
                )

                generate_btn = gr.Button(
                    "🚀 Generate Advertisement",
                    variant="primary",
                    elem_id="generate-btn",
                )

            # ---- RIGHT COLUMN: Outputs ----
            with gr.Column(scale=2, min_width=480):
                gr.Markdown("### 🖼️ Generated Advertisements")

                image_gallery = gr.Gallery(
                    label="Advertisement Variants",
                    columns=2,
                    rows=2,
                    height=480,
                    object_fit="contain",
                    show_label=False,
                    preview=True,
                )

                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2,
                    max_lines=4,
                    elem_id="status-box",
                )

                with gr.Accordion("🔍 Enhanced Prompt (LLM 1 Output)", open=False):
                    enhanced_prompt_output = gr.Textbox(
                        label="Engineered Stable Diffusion Prompt",
                        interactive=False,
                        lines=5,
                        info=(
                            "This is the enriched prompt generated by Qwen2.5-7B-Instruct "
                            "from your advertisement text, used as input to Stable Diffusion XL."
                        ),
                    )

        # -----------------------------------------------------------------
        # Examples
        # -----------------------------------------------------------------
        gr.Markdown("---\n### 💡 Example Prompts — Click to Try")
        gr.Examples(
            examples=[
                ["Nike Air Max — Engineered for champions. Bold, dynamic, built for the streets.", "vibrant", 1],
                ["Luxury Swiss watch — Timeless craftsmanship since 1875. Precision meets elegance.", "luxury", 1],
                ["Fresh organic cold brew coffee — Wake up to something extraordinary.", "minimalist", 2],
                ["Summer beach resort — Escape to paradise. Turquoise waters, white sand, pure bliss.", "cinematic", 1],
                ["Kids educational toy — Fun, colorful, sparks curiosity in every child.", "playful", 2],
            ],
            inputs=[ad_text_input, style_input, num_images_input],
            label="Click any example to populate the form",
        )

        # -----------------------------------------------------------------
        # Pipeline Info Footer
        # -----------------------------------------------------------------
        gr.Markdown("""
---
<div style="text-align:center; color:#718096; font-size:0.82rem; padding:8px 0;">
    <b>Pipeline:</b>
    Ad Text → <b>Qwen2.5-7B-Instruct</b> (Prompt Engineering) →
    <b>Stable Diffusion XL</b> (Image Generation) → Final Advertisement
    &nbsp;|&nbsp;
    <a href="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct" target="_blank">LLM 1</a>
    &nbsp;·&nbsp;
    <a href="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0" target="_blank">LLM 2</a>
</div>
        """)

        # -----------------------------------------------------------------
        # Event Handlers
        # -----------------------------------------------------------------

        # Update style hint when style changes
        def update_style_hint(style):
            desc = STYLE_DESCRIPTIONS.get(style, "")
            return f"💡 *{desc}*"

        style_input.change(
            fn=update_style_hint,
            inputs=[style_input],
            outputs=[style_hint],
        )

        # Generate button click handler
        generate_btn.click(
            fn=generate_ads,
            inputs=[
                ad_text_input,
                style_input,
                num_images_input,
                template_input,
                api_key_input,
            ],
            outputs=[
                image_gallery,
                enhanced_prompt_output,
                status_output,
            ],
            show_progress=True,
        )

    return demo


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
demo = build_ui()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   # Required for HF Spaces
        server_port=7860,         # Default HF Spaces port
        share=False,
        show_error=True,
    )
