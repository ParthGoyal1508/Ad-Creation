"""
main.py
-------
Entry point for Capstone Project CS[03]: Advertisement Creator using Image Generation.

Project Objective:
    Automate the creation of compelling advertisement visuals using two AI models:
    - LLM 1 (Qwen2.5-7B-Instruct): Enhances raw ad text into a rich image generation prompt.
    - LLM 2 (Stable Diffusion XL): Generates the advertisement image from the prompt.

Inputs:
    --ad_text   : Description of the advertisement (e.g., product name, message, target audience)
    --template  : (Optional) Path to an existing ad template image for blending
    --output_dir: Directory to save generated advertisements (default: ./output)
    --hf_api_key: HuggingFace API key (or set via environment variable HF_API_KEY)
    --num_images: Number of advertisement image variants to generate (default: 1)
    --style     : Visual style for the advertisement (default: 'modern')

Output:
    Final advertisement PNG images saved in the specified output directory.

Usage:
    python main.py --ad_text "Nike running shoes - Just Do It" --style "vibrant" --num_images 2
    python main.py --ad_text "Fresh organic coffee" --template ./template.png --output_dir ./ads

Models Used:
    1. Qwen/Qwen2.5-7B-Instruct              — Text LLM for prompt engineering
    2. stabilityai/stable-diffusion-xl-base-1.0 — Image generation model

Justification for Model Selection:
    - Qwen2.5-7B-Instruct is an open-access instruction-tuned LLM from Alibaba that
      supports chat completions on HuggingFace free tier with excellent creative writing
      and prompt engineering capabilities.
    - Stable Diffusion XL is the latest open-source text-to-image model from Stability AI
      that produces high-quality, photorealistic images suitable for commercial advertisements.
    Both models are accessible via the HuggingFace Inference API (free tier available).
"""

import argparse
import os
import sys

from prompt_builder import build_image_prompt
from ad_generator import generate_advertisement

# ---------------------------------------------------------------------------
# Supported advertisement visual styles
# ---------------------------------------------------------------------------
VALID_STYLES = [
    "modern",
    "minimalist",
    "vibrant",
    "luxury",
    "retro",
    "corporate",
    "playful",
    "cinematic"
]

def parse_arguments() -> argparse.Namespace:
    """
    Parses and validates command-line arguments for the advertisement generator.

    Returns:
        argparse.Namespace: Parsed argument values.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Advertisement Creator using AI Image Generation\n"
            "Capstone Project CS[03]\n\n"
            "Generates professional advertisement images using two LLMs:\n"
            "  1. Qwen2.5-7B-Instruct     — enhances ad text into image prompts\n"
            "  2. Stable Diffusion XL     — generates the advertisement image"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Required argument: advertisement text description
    parser.add_argument(
        "--ad_text",
        type=str,
        required=True,
        help="Advertisement description (e.g., 'Nike Air Max - Engineered for champions')"
    )

    # Optional: path to a template image to blend with generated output
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="(Optional) Path to an existing advertisement template image (PNG/JPEG)"
    )

    # Optional: output directory for saving generated advertisements
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save generated advertisement images (default: ./output)"
    )

    # Optional: HuggingFace API key (can also be set via environment variable)
    parser.add_argument(
        "--hf_api_key",
        type=str,
        default=None,
        help="HuggingFace API key (alternatively set via environment variable HF_API_KEY)"
    )

    # Optional: number of advertisement image variants to generate
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="Number of advertisement image variants to generate (default: 1)"
    )

    # Optional: visual style for the advertisement
    parser.add_argument(
        "--style",
        type=str,
        default="modern",
        choices=VALID_STYLES,
        help=f"Visual style for the advertisement. Choices: {VALID_STYLES} (default: modern)"
    )

    return parser.parse_args()

def resolve_api_key(args_key: str) -> str:
    """
    Resolves the HuggingFace API key from CLI argument or environment variable.
    Exits with an error if no key is found.

    Args:
        args_key (str): API key passed via --hf_api_key argument (can be None).

    Returns:
        str: The resolved HuggingFace API key.
    """
    # Priority 1: use key provided directly via argument
    if args_key:
        return args_key

    # Priority 2: check environment variable HF_API_KEY
    env_key = os.environ.get("HF_API_KEY")
    if env_key:
        return env_key

    # No key found — exit with a descriptive error
    print(
        "[Error] HuggingFace API key not found.\n"
        "  Provide it via --hf_api_key <your_key> or set environment variable HF_API_KEY.\n"
        "  Get a free API key at: https://huggingface.co/settings/tokens"
    )
    sys.exit(1)

def validate_inputs(args: argparse.Namespace) -> None:
    """
    Validates user-provided inputs before starting the generation pipeline.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # Validate ad text is not empty
    if not args.ad_text.strip():
        print("[Error] --ad_text cannot be empty.")
        sys.exit(1)

    # Validate template image path if provided
    if args.template and not os.path.exists(args.template):
        print(f"[Error] Template image not found at: {args.template}")
        sys.exit(1)

    # Validate number of images is a positive integer
    if args.num_images < 1:
        print("[Error] --num_images must be at least 1.")
        sys.exit(1)

    if args.num_images > 5:
        print("[Warning] Generating more than 5 images may take a long time with the free API tier.")

def main():
    """
    Main pipeline for the Advertisement Creator:
      Step 1 — Parse and validate CLI arguments
      Step 2 — Resolve HuggingFace API key
      Step 3 — LLM 1: Enhance raw ad text into rich image generation prompt (Qwen2.5-7B-Instruct)
      Step 4 — LLM 2: Generate advertisement images from prompt (Stable Diffusion XL)
      Step 5 — Report saved output file paths
    """

    print("=" * 60)
    print("  Advertisement Creator — Capstone Project CS[03]")
    print("  Powered by Qwen2.5-7B-Instruct + Stable Diffusion XL")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Step 1: Parse and validate arguments
    # -----------------------------------------------------------------------
    args = parse_arguments()
    validate_inputs(args)

    print(f"\n[Config] Advertisement Text : {args.ad_text}")
    print(f"[Config] Visual Style       : {args.style}")
    print(f"[Config] Number of Variants : {args.num_images}")
    print(f"[Config] Output Directory   : {args.output_dir}")
    if args.template:
        print(f"[Config] Template Image     : {args.template}")

    # -----------------------------------------------------------------------
    # Step 2: Resolve API key
    # -----------------------------------------------------------------------
    hf_api_key = resolve_api_key(args.hf_api_key)
    print("\n[Auth] HuggingFace API key resolved successfully.")

    # -----------------------------------------------------------------------
    # Step 3: LLM 1 — Enhance ad text into a detailed image generation prompt
    # -----------------------------------------------------------------------
    print("\n--- Step 1/2: Prompt Engineering (Qwen2.5-7B-Instruct) ---")
    enhanced_prompt = build_image_prompt(
        ad_text=args.ad_text,
        style=args.style,
        hf_api_key=hf_api_key
    )

    # -----------------------------------------------------------------------
    # Step 4: LLM 2 — Generate advertisement images using Stable Diffusion
    # -----------------------------------------------------------------------
    print("\n--- Step 2/2: Image Generation (Stable Diffusion XL) ---")
    saved_files = generate_advertisement(
        image_prompt=enhanced_prompt,
        ad_text=args.ad_text,
        hf_api_key=hf_api_key,
        output_dir=args.output_dir,
        num_images=args.num_images,
        template_path=args.template
    )

    # -----------------------------------------------------------------------
    # Step 5: Summary report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    if saved_files:
        print(f"  Generation Complete! {len(saved_files)} advertisement(s) saved:")
        for path in saved_files:
            print(f"    - {path}")
    else:
        print("  [Warning] No images were generated. Please check your API key and try again.")
    print("=" * 60)

if __name__ == "__main__":
    main()
