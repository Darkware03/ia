import logging
import os
import re
import json
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("token-brief-api-local")

# Config
MODEL_ID = os.getenv("MODEL_ID", "google/gemma-7b-it")
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
    A partir del siguiente texto, genera un token de memecoin con los siguientes campos separados por `|`:

    Texto:
    \"\"\"
    {base_text}
    \"\"\"

    Devuelve exactamente una sola línea con los siguientes campos, en este orden, separados por `|`:

    name | symbol | description_short | description_long | hashtags_separados_por_coma | emojis_separados_por_coma | image_prompt | disclaimers_separados_por_coma

    Ejemplo de formato:
    el_bufon | BUFON | Meme político viral | Crítica satírica sobre el poder | #humor,#politica | 😂🔥 | Meme de un político en estilo arte pop | No es consejo financiero,Solo entretenimiento

    ❗ No escribas encabezados, ni explicaciones, ni saltos de línea. Solo el contenido de una sola línea separado por `|`. ❗
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    try:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50
            )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info("🧪 RAW GENERATED TEXT:\n%s", generated_text)

        # Buscar línea con exactamente 8 pipes (|), es decir 9 partes
        for line in generated_text.splitlines():
            if line.count("|") == 8:
                parts = [part.strip() for part in line.split("|")]
                if len(parts) == 9:
                    return JSONResponse(content={
                        "name": parts[0],
                        "symbol": parts[1],
                        "description_short": parts[2],
                        "description_long": parts[3],
                        "hashtags": parts[4].split(","),
                        "emojis": parts[5].split(","),
                        "image_prompt": parts[6],
                        "disclaimers": parts[7].split(",")
                    })

        raise ValueError("No se encontró una línea válida con separadores |")

    except Exception as e:
        logger.exception("❌ Error procesando la solicitud:")
        return JSONResponse(
            status_code=502,
            content={
                "error": True,
                "status_code": 502,
                "detail": {
                    "message": "El modelo no devolvió una línea válida con separadores |",
                    "raw": generated_text if 'generated_text' in locals() else ""
                }
            }
        )

