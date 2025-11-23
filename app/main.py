import io
import os
import random
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

from app.models import get_captioner, get_summarizer, get_sd_pipeline, image_to_base64, apply_style, generate_elaboration

app = FastAPI(title="Vision Studio", version="0.1.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

app.mount("/static", StaticFiles(directory=os.path.join(PROJECT_ROOT, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(PROJECT_ROOT, "templates"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/caption")
async def caption_image(
    file: UploadFile = File(...),
    max_new_tokens: int = Form(32),
    temperature: float = Form(1.0),
    top_p: float = Form(0.9),
    repetition_penalty: float = Form(1.0),
    prefix: str = Form(""),
    suffix: str = Form(""),
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        cap = get_captioner()
        # Only pass supported kwargs to the image-to-text pipeline
        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens),
        }
        outputs = cap(image, **gen_kwargs)
        text = outputs[0]["generated_text"].strip()
        if prefix:
            text = f"{prefix} {text}"
        if suffix:
            text = f"{text} {suffix}"
        return {"caption": text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/txt2img")
async def txt2img(
    request: Request,
):
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        negative_prompt = data.get("negative_prompt", "")
        style = data.get("style", "photographic")
        steps = int(data.get("steps", 30))
        guidance = float(data.get("guidance", 8.0))
        seed = data.get("seed")
        width = int(data.get("width", 512))
        height = int(data.get("height", 512))
        if not prompt:
            return JSONResponse(status_code=400, content={"error": "Prompt is required"})

        full_prompt = apply_style(prompt, style)
        pipe = get_sd_pipeline()
        if seed is None or seed == "":
            seed = random.randint(0, 2**32 - 1)
        import torch
        # Use CUDA generator only on CUDA; on CPU and MPS, use CPU generator
        gen_device = "cuda" if (hasattr(pipe, "device") and getattr(pipe.device, "type", "cpu") == "cuda") else "cpu"
        torch_gen = torch.Generator(device=gen_device).manual_seed(int(seed))

        # Strong negative prompt fallback to enforce realism if none provided
        neg = negative_prompt or (
            "cartoon, anime, drawing, painting, illustration, cgi, 3d render, low quality, lowres, blurry, oversaturated, noisy, artifacts, deformed, bad anatomy, extra limbs, extra fingers, watermark, text"
        )

        image = pipe(
            prompt=full_prompt,
            negative_prompt=neg,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=torch_gen,
        ).images[0]
        b64 = image_to_base64(image)
        return {"image_base64": b64, "seed": int(seed)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/summarize")
async def summarize(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        max_len = int(data.get("max_length", 130))
        min_len = int(data.get("min_length", 30))
        if not text.strip():
            return JSONResponse(status_code=400, content={"error": "Text is required"})
        summarizer = get_summarizer()
        out = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
        return {"summary": out[0]["summary_text"]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/elaborate")
async def elaborate(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        tone = data.get("tone", "neutral")
        length = data.get("length", "medium")
        creativity = float(data.get("creativity", 0.7))
        if not text.strip():
            return JSONResponse(status_code=400, content={"error": "Text is required"})
        elaborated = generate_elaboration(text, tone=tone, length=length, creativity=creativity)
        return {"elaboration": elaborated}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)
