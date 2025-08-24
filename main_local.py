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

    # === Prompt ultra estricto (7 campos, 6 comas) ===
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
                max_new_tokens=120,      # suficiente para 1 línea
                do_sample=True,
                temperature=0.5,         # más obediente
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.05  # evita repetir el prompt
            )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info("🧪 RAW GENERATED TEXT:\n%s", generated_text)

        import re

        # --- Utilidad: limpieza mínima ---
        def clean(s: str) -> str:
            return re.sub(r"\s+", " ", s).strip().strip("`* ")

        # 1) Preferencia: <csv> ... </csv>
        m = re.search(r"<csv>(.*?)</csv>", generated_text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            line = clean(m.group(1))
            # Validar: 6 comas (=> 7 campos) y sin pipes
            if line.count(",") == 6 and "|" not in line:
                return JSONResponse(content={"success": True, "token": line})

        # 2) Fallback: primera línea con exactamente 6 comas y sin pipes
        for raw_line in generated_text.splitlines():
            line = clean(raw_line)
            if line.count(",") == 6 and "|" not in line and not line.lower().startswith(("name , symbol", "name, symbol")):
                return JSONResponse(content={"success": True, "token": line})

        # 3) Fallback: si vino en pipes con 6 pipes, conviértelo a comas
        for raw_line in generated_text.splitlines():
            line = clean(raw_line)
            if line.count("|") == 6:
                csv_line = clean(line.replace("|", ","))
                # Re-validar comas
                if csv_line.count(",") == 6:
                    return JSONResponse(content={"success": True, "token": csv_line})

        # 4) Fallback: reconstrucción desde bullets name:, symbol:, ...
        #    Captura tipo: "name: algo", case-insensitive
        fields = ["name", "symbol", "description_short", "description_long",
                  "hashtags_separados_por_coma", "emojis_separados_por_coma",
                  "disclaimers_separados_por_coma"]
        found = {}
        for f in fields:
            # Busca 'f: valor' hasta fin de línea
            mm = re.search(rf"{f}\s*:\s*(.+)", generated_text, flags=re.IGNORECASE)
            if mm:
                # corta en fin de línea y limpia
                val = clean(mm.group(1).splitlines()[0])
                # elimina posibles separadores conflictivos
                val = val.replace("|", " ").replace("\n", " ").strip()
                found[f] = val

        if found:
            row = []
            for f in fields:
                row.append(found.get(f, ""))  # vacío si no está
            csv_line = ",".join(row)
            # Asegura 6 comas
            if csv_line.count(",") == 6:
                return JSONResponse(content={"success": True, "token": csv_line})

        # 5) Último recurso: responde éxito=false con el raw (sin 5xx)
        return JSONResponse(
            status_code=200,
            content={
                "success": False,
                "message": "No se pudo extraer una línea CSV válida (7 campos).",
                "raw": generated_text
            }
        )

    except Exception:
        logger.exception("❌ Error procesando la solicitud:")
        return JSONResponse(
            status_code=200,  # evita 5xx, para que tu JS no truene
            content={
                "success": False,
                "message": "Error inesperado en el servidor de generación.",
                "raw": generated_text if 'generated_text' in locals() else ""
            }
        )
