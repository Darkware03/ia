import logging
import os
import re
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
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

    prompt = f"""
    A partir del siguiente texto, genera un token de memecoin original.

    Debes responder únicamente con un JSON válido en este formato:
    {{
      "name": "...",
      "symbol": "...",
      "description_short": "...",
      "description_long": "...",
      "hashtags": ["...", "..."],
      "emojis": ["...", "..."],
      "image_prompt": "...",
      "disclaimers": ["No es consejo financiero", "Solo para entretenimiento"]
    }}

    Texto:
    \"\"\"
    {base_text}
    \"\"\"

    Responde solo con el JSON. No incluyas explicación, introducción ni formato de ejemplo.
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

        # Extraer JSON usando regex
        match = re.search(r"\{[\s\S]*?\}", generated_text)
        if not match:
            raise ValueError("El modelo no devolvió JSON válido.")

        response_json = json.loads(match.group(0))
        return JSONResponse(content=response_json)

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
