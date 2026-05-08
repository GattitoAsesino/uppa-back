from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from especie_detector.main import detectar_especie

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
    resultado = detectar_especie(image_bytes)
    return resultado