import logging
import os
import re
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("token-brief-api-local")

# Config
MODEL_ID = os.getenv("MODEL_ID", "google/gemma-2b-it")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32

# Modelo
logger.info(f"Cargando modelo: {MODEL_ID} (device={DEVICE}, dtype={DTYPE})")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=DTYPE
)

# FastAPI
app = FastAPI()


# Input schema
class GenerationRequest(BaseModel):
    text: str
    language: str = "es"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
async def generate(request: GenerationRequest):
    base_text = request.text.strip()

    # PROMPT optimizado para forzar JSON válido
    prompt = f"""Texto:
\"\"\"
{base_text}
\"\"\"

Devuelve únicamente un JSON válido con los siguientes campos completados según el texto anterior:

{{
  "name": "Nombre original del token",
  "symbol": "Símbolo corto (3–6 letras)",
  "description_short": "Resumen divertido y viral",
  "description_long": "Descripción completa con tono de humor, sátira o crítica social",
  "hashtags": ["#ejemplo1", "#ejemplo2"],
  "emojis": ["🔥", "💰"],
  "image_prompt": "Prompt para IA para generar una imagen del token",
  "disclaimers": ["No es consejo financiero", "Solo para entretenimiento"]
}}

Solo devuelve el JSON. No incluyas ninguna explicación, encabezado ni código de ejemplo.
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    try:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50
            )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info("🧪 RAW GENERATED TEXT:\n%s", generated_text)

      

        return JSONResponse(content=generated_text)

    except Exception as e:
        logger.exception("❌ Error procesando la solicitud:")
        return JSONResponse(
            status_code=502,
            content={
                "error": True,
                "status_code": 502,
                "detail": {
                    "message": "El modelo no devolvió JSON válido.",
                    "raw": generated_text if 'generated_text' in locals() else ""
                }
            }
        )
