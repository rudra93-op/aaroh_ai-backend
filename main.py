from fastapi import FastAPI, UploadFile, File
import shutil
import os
from chord_engine import analyze_audio

app = FastAPI()


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    temp_path = f"temp_audio{ext}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = analyze_audio(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return result
