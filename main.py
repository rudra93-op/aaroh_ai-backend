# main.py
from fastapi import FastAPI, UploadFile, File, Form
import shutil
import os
import librosa
import numpy as np

from chord_engine import analyze_audio

app = FastAPI()

# In-memory state (replace with Redis later)
SONG_STATE = {
    "bpm": None,
    "chords": []
}


@app.get("/health")
def health():
    return {"status": "ok"}


# ===============================
# 🎵 UPLOAD SONG (ONCE)
# ===============================
@app.post("/upload")
async def upload_song(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1]
    temp_path = f"song{ext}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = analyze_audio(temp_path)
        SONG_STATE["bpm"] = result["bpm"]
        SONG_STATE["chords"] = result["chords"]

        return {
            "status": "song_loaded",
            "bpm": result["bpm"],
            "chord_count": len(result["chords"])
        }
    finally:
        os.remove(temp_path)


# ===============================
# 🎸 REALTIME CHUNK FEEDBACK
# ===============================
@app.post("/realtime/chunk")
async def realtime_chunk(
    audio: UploadFile = File(...),
    timestamp: float = Form(...)
):
    # Find expected chord
    expected = None
    for c in SONG_STATE["chords"]:
        if c["start"] <= timestamp <= c["end"]:
            expected = c
            break

    if expected is None or expected["chord"] == "N" or expected["chroma"] is None:
        return {
            "expected_chord": None,
            "confidence": 0.0,
            "is_correct": False
        }

    chunk_path = "chunk.wav"
    with open(chunk_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    try:
        y, sr = librosa.load(chunk_path, sr=44100, mono=True)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        user_chroma = chroma.mean(axis=1)
        user_chroma /= (np.linalg.norm(user_chroma) + 1e-6)

        ref_chroma = np.array(expected["chroma"])
        confidence = float(np.dot(user_chroma, ref_chroma))

        return {
            "expected_chord": expected["chord"],
            "confidence": round(confidence, 2),
            "is_correct": confidence >= 0.7
        }
    finally:
        os.remove(chunk_path)
