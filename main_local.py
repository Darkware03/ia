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
### INSTRUCCIONES (LEE Y OBEDECE)
Genera EXACTAMENTE UNA SOLA LÍNEA en formato CSV con **7** campos en este orden:
name,symbol,description_short,description_long,hashtags_separados_por_coma,emojis_separados_por_coma,disclaimers_separados_por_coma

REGLAS OBLIGATORIAS:
- Usa coma "," como separador de campos.
- NO uses comillas, NO uses barras verticales "|", NO uses saltos de línea extra, NO agregues explicaciones.
- Si un campo no aplica, deja el campo vacío, pero conserva las comas.
- DEVUELVE ÚNICAMENTE esa línea CSV **envuelta** entre las etiquetas <csv> y </csv>.
- No devuelvas nada más fuera de esas etiquetas.

### TEXTO
\"\"\" 
{base_text}
\"\"\"

### RESPUESTA
<csv>name,symbol,description_short,description_long,hashtags,emojis,disclaimers</csv>
""".strip()

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    try:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.05
            )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info("🧪 RAW GENERATED TEXT:\n%s", generated_text)

        import re

        def clean(s: str) -> str:
            return re.sub(r"\s+", " ", s).strip().strip("`* ")

        # 1) patrón principal: tomar la PRIMERA línea no vacía DESPUÉS de </csv>
        m = re.search(r"</csv>(.*)$", generated_text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            tail = m.group(1)
            for raw in tail.splitlines():
                line = raw.strip()
                if not line:
                    continue
                # ignorar títulos/markdown si aparecieran
                if line.startswith("#") or line.lower().startswith("respuesta") or line.startswith("---"):
                    continue
                # esta es la línea que quieres (ej: "John Doe,,I’m super..., ...")
                return JSONResponse(content={"success": True, "token": line})

        # 2) fallback: si no hubo </csv>, buscar la primera línea con 6 comas (7 campos)
        for raw in generated_text.splitlines():
            line = clean(raw)
            if line.count(",") == 6 and "|" not in line:
                return JSONResponse(content={"success": True, "token": line})

        # 3) fallback: si vino con pipes y hay 6 pipes, convierte a comas
        for raw in generated_text.splitlines():
            line = clean(raw)
            if line.count("|") == 6:
                csv_line = clean(line.replace("|", ","))
                if csv_line.count(",") == 6:
                    return JSONResponse(content={"success": True, "token": csv_line})

        # 4) si nada, devuelve raw para inspección (sin 5xx)
        return JSONResponse(
            status_code=200,
            content={
                "success": False,
                "message": "No se encontró una línea CSV válida después de </csv>.",
                "raw": generated_text
            }
        )

    except Exception:
        logger.exception("❌ Error procesando la solicitud:")
        return JSONResponse(
            status_code=200,
            content={
                "success": False,
                "message": "Error inesperado en el servidor de generación.",
                "raw": generated_text if 'generated_text' in locals() else ""
            }
        )
