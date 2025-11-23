import base64
import io
import os
from functools import lru_cache
from typing import Optional

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


def device_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    elif torch.backends.mps.is_available():
        # MPS works best with float16 disabled sometimes; use float32
        return torch.device("mps"), torch.float32
    else:
        return torch.device("cpu"), torch.float32


@lru_cache(maxsize=1)
def get_captioner():
    # ViT-GPT2 image captioning
    return pipeline(
        "image-to-text",
        model="nlpconnect/vit-gpt2-image-captioning",
        device=0 if torch.cuda.is_available() else -1,
    )


@lru_cache(maxsize=1)
def get_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)


@lru_cache(maxsize=1)
def get_t5():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


@lru_cache(maxsize=1)
def get_sd_pipeline():
    device, dtype = device_dtype()
    model_id = os.getenv("SD_MODEL", "runwayml/stable-diffusion-v1-5")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if (dtype==torch.float16 and device.type=="cuda") else torch.float32,
        safety_checker=None,
    )
    try:
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    except Exception:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass
    try:
        if device.type == "cuda":
            pipe = pipe.to(device)
        elif device.type == "mps":
            pipe = pipe.to(device)
        else:
            pipe = pipe.to("cpu")
    except Exception:
        pipe = pipe.to("cpu")
    return pipe


def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def apply_style(prompt: str, style: str) -> str:
    styles = {
        "none": "",
        "photographic": "photorealistic, natural skin texture, 50mm lens, shallow depth of field, bokeh, HDR, high detail, sharp focus, soft lighting",
        "cinematic": "cinematic lighting, film still, dramatic shadows, volumetric light",
        "anime": "anime style, clean lines, vibrant colors, studio ghibli, makoto shinkai",
        "watercolor": "watercolor painting, soft brush strokes, pastel tones",
        "3d": "3D render, octane render, highly detailed, global illumination",
        "pixel": "pixel art, 16-bit, retro style, crisp pixels",
    }
    suffix = styles.get(style or "none", "")
    if suffix:
        return f"{prompt}, {suffix}"
    return prompt


def generate_elaboration(text: str, tone: str = "neutral", length: str = "medium", creativity: float = 0.7, max_new_tokens: int = 256) -> str:
    tokenizer, model = get_t5()
    length_map = {"short": "in about 3-4 sentences", "medium": "in about 6-8 sentences", "long": "in 10-12 sentences"}
    instruction = f"Rewrite and expand the following text {length_map.get(length, '')} with a {tone} tone, add examples if helpful:\n\n{text}"
    inputs = tokenizer(instruction, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=max(0.1, float(creativity)),
            top_p=0.9,
            max_new_tokens=max_new_tokens,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
