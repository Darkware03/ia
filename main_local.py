import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("token-brief-api-local")

# Configuración
MODEL_ID = "google/gemma-2b-it"
MAX_NEW_TOKENS = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Inicialización del modelo
logger.info(f"Cargando modelo: {MODEL_ID} (device={DEVICE}, dtype={DTYPE})")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token  # Fix para modelos como Gemma
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto"
)

# FastAPI
app = FastAPI()

class TokenRequest(BaseModel):
    text: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/generate")
async def generate(request: TokenRequest):
    try:
        prompt = f"""
        Genera un token de memecoin en formato JSON. Llena cada campo con información creativa, divertida o relevante según el texto base. No dejes campos vacíos.

        Formato:
        {{
          "name": "Nombre del token",
          "symbol": "Símbolo corto (3–6 letras)",
          "description_short": "Resumen atractivo en una línea",
          "description_long": "Descripción completa de la idea del token",
          "hashtags": ["#ejemplo", "#token"],
          "emojis": ["🔥", "💰"],
          "image_prompt": "Prompt para generar una imagen con IA del token",
          "disclaimers": ["No es consejo financiero", "Solo para entretenimiento"]
        }}

        Texto base:
        \"\"\"
        {request.text}
        \"\"\"

        Responde SOLO con el JSON válido.
        """

        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        logger.info("🧪 RAW GENERATED TEXT:\n%s", generated_text)

        # Extraer JSON desde la respuesta
        start_idx = generated_text.find("{")
        end_idx = generated_text.rfind("}") + 1
        json_text = generated_text[start_idx:end_idx]

        try:
            response = eval(json_text, {}, {})  # Reemplazar con json.loads si necesario
            return JSONResponse(content=response)
        except Exception as e:
            return JSONResponse(
                status_code=502,
                content={"error": True, "status_code": 502, "detail": {
                    "message": "El modelo no devolvió JSON válido.",
                    "raw": generated_text
                }},
            )

    except Exception as e:
        logger.exception("Error al generar texto")
        return JSONResponse(
            status_code=500,
            content={"error": True, "status_code": 500, "detail": str(e)},
        )
