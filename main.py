from fastapi import FastAPI, UploadFile, File
import shutil
from chord_engine import analyze_audio

app = FastAPI()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    with open("temp.wav", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = analyze_audio("temp.wav")
    return result
