from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from especie_detector.main import detectar_especie
from enfermedad_detector.main  import detectar_salud

app = FastAPI(title="Úppa API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def root():
    return {"mensaje": "Úppa API funcionando"}

@app.post("/analizar")
async def analizar(file: UploadFile = File(...)):
    image_bytes = await file.read()

    especie = detectar_especie(image_bytes)
    salud   = detectar_salud(image_bytes)

    return {
        # Especie
        "detectado":      especie["detectado"],
        "especie":        especie.get("especie"),
        "confianza":      especie.get("confianza"),
        "id_especie":     especie.get("id_especie"),   # ← agregado
        "cultivada":      especie.get("cultivada"),    # ← agregado
        # Salud
        "sano":            salud["sano"],
        "estado":          salud.get("estado"),
        "confianza_salud": salud.get("confianza_salud"),
        # Mensajes
        "mensaje_especie": especie.get("mensaje"),
        "mensaje_salud":   salud.get("mensaje_salud")
    }