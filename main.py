from fastapi import FastAPI, UploadFile, File
import shutil
from chord_engine import predict_chords

app = FastAPI()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    with open("temp.wav", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    chords = predict_chords("temp.wav")
    return {"chords": chords}
