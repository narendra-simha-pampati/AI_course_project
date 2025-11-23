# AI Studio (Local Transformers)

A clean FastAPI web app with four features running on local Hugging Face models (no external APIs):

- Image captioning with adjustable parameters and editable prefix/suffix
- Text to Image with personalization (style presets, negative prompt, steps, guidance, seed, size)
- Text summarization
- Text elaboration (tone, length, creativity)

## Requirements

- Python 3.10+
- Internet on first run to download models
- GPU optional (CUDA/MPS). CPU works but is slower for text-to-image.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000

## Environment

- To switch Stable Diffusion model:
  ```bash
  export SD_MODEL=stabilityai/stable-diffusion-2-1
  ```

## Notes

- First request for each feature will be slow while models load and download.
- Text-to-image on CPU can take 1-3 minutes. Reduce steps, size, or use a GPU for faster results.
- No external APIs are used; all models run locally via Transformers/Diffusers.
