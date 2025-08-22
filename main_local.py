import os
import re
import json
import logging
from typing import Optional, Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# =======================
# Logging
# =======================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("token-brief-api-local")

# =======================
# .env
# =======================
load_dotenv()
MODEL_ID = os.getenv("MODEL_ID", "mistralai/Mistral-7B-Instruct").strip()
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "600"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
HOST = os.getenv("HOST", "0.0.0.0").strip()
PORT = int(os.getenv("PORT", "3010"))

SYSTEM_PROMPT = """Eres un generador de briefs de tokens para Pump.fun.
Devuelve SOLO un JSON válido con este esquema EXACTO:

{
  "name": "",
  "symbol": "",
  "description_short": "",
  "description_long": "",
  "hashtags": [],
  "emojis": [],
  "image_prompt": "",
  "disclaimers": []
}

Reglas:
- "symbol": 3-6 letras, MAYÚSCULAS, sin signos.
- "description_short": <= 280 caracteres.
- Tono respetuoso si hay personas reales; añade disclaimers como:
  "Token no oficial. Sin relación con personas/entidades reales.",
  "Solo con fines de entretenimiento. No es consejo financiero."
- Responde SOLO el JSON. Nada más.
"""

# =======================
# Modelos Pydantic
# =======================
class GenerateIn(BaseModel):
    text: str = Field(..., description="Texto base a analizar")
    language: str = Field("es", description="Idioma de salida (es/en)")

class TokenBrief(BaseModel):
    name: str
    symbol: str
    description_short: str
    description_long: str
    hashtags: list
    emojis: list
    image_prompt: str
    disclaimers: list

    @validator("symbol")
    def symbol_rules(cls, v):
        if not re.fullmatch(r"[A-Z]{3,6}", v or ""):
            raise ValueError("symbol debe ser 3-6 letras MAYÚSCULAS (A-Z) sin signos.")
        return v

    @validator("description_short")
    def short_len(cls, v):
        return (v or "")[:280]

# =======================
# Carga del modelo local
# =======================
def _get_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    # CPU
    return "cpu", torch.float32

device, dtype = _get_device_and_dtype()
log.info("Cargando modelo: %s (device=%s, dtype=%s)", MODEL_ID, device, dtype)

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None
    )
    if device == "cpu":
        model.to(device)
except Exception as e:
    log.exception("Error cargando el modelo local")
    raise

# =======================
# Utilidades generación
# =======================
def build_user_prompt(user_text: str, language: str = "es") -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Texto base ({language}):\n\"\"\"{user_text}\"\"\"\n\n"
        f"Devuelve SOLO el JSON válido."
    )

def extract_json_block(text: str) -> Optional[dict]:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    candidate = m.group(0)
    try:
        return json.loads(candidate)
    except Exception:
        return None

def generate_locally(user_text: str, language: str = "es") -> str:
    # Mensajes en formato chat (la mayoría de instruct ya traen plantilla)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(user_text, language)}
    ]
    # apply_chat_template crea el prompt correcto para cada modelo
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    ).to(model.device)

    # Parámetros de generación
    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    # Decodificamos sólo lo generado nuevo
    gen_tokens = outputs[0][inputs.shape[-1]:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return text

# =======================
# FastAPI
# =======================
app = FastAPI(title="Token Brief API (local transformers)", version="1.0.0")

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": True, "status_code": exc.status_code, "detail": exc.detail},
    )

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.exception("Uncaught error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": True, "status_code": 500, "detail": "Error interno del servidor"},
    )

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID, "device": device}

@app.post("/generate", response_model=TokenBrief)
async def generate(payload: GenerateIn):
    try:
        raw = generate_locally(payload.text, payload.language)
    except RuntimeError as rt:
        # Error típico: OOM (falta VRAM/RAM)
        raise HTTPException(status_code=500, detail=f"Error de ejecución (posible falta de memoria): {rt}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    parsed = extract_json_block(raw)
    if parsed is None:
        # Intento directo si el modelo devolvió JSON “limpio”
        try:
            parsed = json.loads(raw)
        except Exception:
            raise HTTPException(status_code=502, detail={"message": "El modelo no devolvió JSON válido.", "raw": raw[:1000]})

    # Normalizaciones mínimas
    parsed.setdefault("name", "")
    parsed.setdefault("symbol", "TOKEN")
    parsed.setdefault("description_short", "")
    parsed.setdefault("description_long", "")
    parsed.setdefault("hashtags", [])
    parsed.setdefault("emojis", [])
    parsed.setdefault("image_prompt", "")
    parsed.setdefault("disclaimers", [])

    # Validación/Corrección de symbol si hace falta
    try:
        result = TokenBrief(**parsed)
    except Exception:
        parsed["symbol"] = re.sub("[^A-Za-z]", "", parsed.get("symbol", "")).upper()[:6]
        if len(parsed["symbol"]) < 3:
            parsed["symbol"] = (parsed["symbol"] + "TOKEN")[:6]
        result = TokenBrief(**parsed)

    return result
