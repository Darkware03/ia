import logging
import os
import re
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("token-brief-api-local")

# Modelo y dispositivo
MODEL_ID = os.getenv("MODEL_ID", "google/gemma-7b-it")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32

# Cargar tokenizer y modelo
logger.info(f"Cargando modelo: {MODEL_ID} (device={DEVICE}, dtype={DTYPE})")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=DTYPE
)

# Crear la app FastAPI
app = FastAPI()

# Esquema de entrada
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

    Devuelve exactamente una sola línea con los siguientes campos es para crear tun token de memecoin, 
   ❗ No escribas encabezados, ni explicaciones, ni saltos de línea❗
        en este orden:

    name , symbol , description_short , description_long , hashtags_separados_por_coma , emojis_separados_por_coma ,  disclaimers_separados_por_coma
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

        # --- Utilidades de limpieza/chequeo ---
        def normalize(s: str) -> str:
            # quita asteriscos/backticks, dobles espacios, y lower-case para comparar
            s2 = s.replace("`", " ").replace("*", " ")
            s2 = re.sub(r"\s+", " ", s2).strip().lower()
            return s2

        HEADER_CANON = normalize(
            "name | symbol | description_short | description_long | "
            "hashtags_separados_por_coma | emojis_separados_por_coma | "
            "image_prompt | disclaimers_separados_por_coma"
        )

        def is_header_line(line: str) -> bool:
            n = normalize(line)
            if HEADER_CANON in n:
                return True
            if n.startswith("ejemplo de formato"):
                return True
            if n.startswith("name | symbol"):
                return True
            return False

        def looks_like_token_line(line: str) -> bool:
            l = line.strip(" `*")
            return (l.count("|") == 8) and (not is_header_line(l))

        # --- 1) Preferencia: línea justo después de "Token:" ---
        m = re.search(r"token\s*:\s*\n\s*(.+)", generated_text, flags=re.IGNORECASE)
        if m:
            after_token = m.group(1).strip()
            # Puede que la línea posterior tenga más texto; separar hasta el fin de línea
            first_line = after_token.splitlines()[0].strip(" `*")
            if looks_like_token_line(first_line):
                return JSONResponse(content={"success": True, "token": first_line})

        # --- 2) Fallback: primera línea en TODO el texto con 8 pipes, excluyendo encabezados ---
        for raw_line in generated_text.splitlines():
            line = raw_line.strip(" `*")
            if looks_like_token_line(line):
                return JSONResponse(content={"success": True, "token": line})

        # --- 3) Si nada matchea, devolvemos el raw para inspección (pero sin error 5xx) ---
        return JSONResponse(
            status_code=200,
            content={
                "success": False,
                "message": "No se encontró una línea válida con separadores `|`.",
                "raw": generated_text
            }
        )

    except Exception:
        logger.exception("❌ Error procesando la solicitud:")
        return JSONResponse(
            status_code=502,
            content={
                "error": True,
                "status_code": 502,
                "detail": {
                    "message": "Ocurrió un error inesperado.",
                    "raw": generated_text if 'generated_text' in locals() else ""
                }
            }
        )
