---
title: AI Advertisement Creator
emoji: 🎨
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
short_description: AI ad generator using Qwen2.5 + SDXL
---

# 🎨 AI Advertisement Creator

**Capstone Project CS[03]** — Automated advertisement image generation using a two-stage AI pipeline.

## 🚀 How It Works

```
Your Ad Text
    │
    ▼
┌─────────────────────────────┐
│  LLM 1: Qwen2.5-7B-Instruct │  ← Transforms raw text into a rich
│  (Prompt Engineering)        │    Stable Diffusion image prompt
└─────────────────────────────┘
    │
    ▼ Enhanced Prompt
┌─────────────────────────────┐
│  LLM 2: Stable Diffusion XL │  ← Generates 1024×1024 advertisement
│  (Image Generation)          │    images from the engineered prompt
└─────────────────────────────┘
    │
    ▼
Final Advertisement Image (PNG)
  + Ad text overlay banner
  + Optional brand template blend
```

## 🎛️ Features

- **8 Visual Styles** — modern, minimalist, vibrant, luxury, retro, corporate, playful, cinematic
- **Multi-variant generation** — generate up to 4 advertisement variants in one run
- **Template blending** — upload your brand template to blend with AI-generated images
- **Transparent pipeline** — see the engineered prompt used for image generation
- **One-click examples** — try pre-built prompts to explore the tool instantly

## 🔑 Setup

### Option 1: Use the Space directly
1. Enter your [HuggingFace API key](https://huggingface.co/settings/tokens) in the API Key field
2. Enter your advertisement text
3. Choose a visual style and click **Generate**

### Option 2: Set API Key as a Secret (for Space owners)
In your Space settings → **Secrets** → add `HF_API_KEY` with your token.
The app will automatically pick it up — no key input needed in the UI.

## 🏃 Run Locally

```bash
git clone <your-repo-url>
cd Codebase
pip install -r requirements.txt
export HF_API_KEY=hf_your_token_here
python app.py
# Open http://localhost:7860
```

## 📦 Tech Stack

| Component | Technology |
|---|---|
| UI Framework | Gradio 4.x |
| Prompt Engineering LLM | Qwen/Qwen2.5-7B-Instruct |
| Image Generation | stabilityai/stable-diffusion-xl-base-1.0 |
| Image Processing | Pillow (PIL) |
| API Layer | HuggingFace InferenceClient |
| Deployment | Hugging Face Spaces |

## ⚠️ Notes

- **Free tier models** may show a `503 Loading` error on first call — wait 20–30 seconds and retry
- Image generation takes **30–90 seconds** per image on the free API tier
- Maximum **4 variants** per generation to stay within free API limits

## 📄 License

MIT — Free to use and modify for educational purposes.
