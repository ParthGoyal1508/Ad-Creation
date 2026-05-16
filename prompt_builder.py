"""
prompt_builder.py
-----------------
LLM 1: Uses HuggingFace InferenceClient with Qwen2.5-7B-Instruct to transform
raw advertisement text into a rich, descriptive image-generation prompt
suitable for Stable Diffusion XL.
Uses chat completions endpoint (conversational task).
"""

from huggingface_hub import InferenceClient

# Model used for prompt engineering via chat completions
# Qwen2.5-7B-Instruct is open-access and supports chat completions on HF free tier
HF_TEXT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

def build_image_prompt(ad_text: str, style: str, hf_api_key: str) -> str:
    """
    Converts raw advertisement text into a detailed image generation prompt
    using the Qwen2.5-7B-Instruct LLM via HuggingFace Inference API.

    Args:
        ad_text  (str): Raw advertisement description provided by the user.
        style    (str): Desired visual style (e.g., 'modern', 'minimalist', 'vibrant').
        hf_api_key (str): HuggingFace API key for authentication.

    Returns:
        str: An enhanced, detailed prompt ready for image generation.
    """

    print(f"[LLM 1 - Qwen2.5-7B] Enhancing advertisement text into image prompt...")

    # Build chat messages for the conversational prompt
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert advertising creative director and AI image prompt engineer. "
                "Convert the advertisement idea into a detailed, vivid, high-quality image "
                "generation prompt for Stable Diffusion XL. Describe the scene, lighting, "
                "composition, colors, style, and mood in under 150 words. "
                "Output ONLY the image prompt, nothing else."
            )
        },
        {
            "role": "user",
            "content": (
                f"Advertisement Text: {ad_text}\n"
                f"Visual Style: {style}\n\n"
                f"Generate a detailed, photorealistic Stable Diffusion image prompt."
            )
        }
    ]

    try:
        # InferenceClient — chat completions supported by Qwen2.5-7B-Instruct
        client = InferenceClient(api_key=hf_api_key)

        response = client.chat.completions.create(
            model=HF_TEXT_MODEL,
            messages=messages,
            max_tokens=200,
            temperature=0.7,
            top_p=0.9
        )

        # Extract the generated text from the response
        enhanced_prompt = response.choices[0].message.content.strip()

        # Fallback: if LLM returns empty, use a styled version of the original text
        if not enhanced_prompt:
            enhanced_prompt = _fallback_prompt(ad_text, style)

        print(f"[LLM 1 - Qwen2.5-7B] Enhanced prompt: {enhanced_prompt[:100]}...")
        return enhanced_prompt

    except Exception as e:
        # On API failure, fall back to a template-based prompt
        print(f"[LLM 1 - Qwen2.5-7B] API call failed: {e}. Using fallback prompt.")
        return _fallback_prompt(ad_text, style)

def _fallback_prompt(ad_text: str, style: str) -> str:
    """
    Generates a basic fallback prompt when the LLM API is unavailable.

    Args:
        ad_text (str): Raw advertisement description.
        style   (str): Desired visual style.

    Returns:
        str: A simple but usable image generation prompt.
    """
    return (
        f"A {style} professional advertisement image showcasing: {ad_text}. "
        f"High quality, photorealistic, studio lighting, clean composition, "
        f"vibrant colors, commercial photography style, 8k resolution."
    )
