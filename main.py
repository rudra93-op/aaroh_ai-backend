# main.py
from fastapi import FastAPI, UploadFile, File, Form
import shutil
import os
import numpy as np
import librosa

from chord_engine import analyze_audio

app = FastAPI()

# ===============================
# In-memory state (later Redis)
# ===============================

SONG_STATE = {
    "bpm": None,
    "chords": []
}

# ===============================
# Health (Render / uptime)
# ===============================

@app.get("/health")
@app.head("/health")
def health():
    return {"status": "ok"}


# ===============================
# Upload song
# ===============================

@app.post("/upload")
async def upload_song(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1]
    temp_path = f"temp_song{ext}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = analyze_audio(temp_path)

        SONG_STATE["bpm"] = result["bpm"]
        SONG_STATE["chords"] = result["chords"]

        return {
            "status": "song_loaded",
            "bpm": result["bpm"],
            "chord_count": len(result["chords"]),
            "chords": result["chords"]  # ✅ YOU SEE EVERYTHING
        }

    finally:
        os.remove(temp_path)


# ===============================
# Get song data (Playground UI)
# ===============================

@app.get("/song/state")
def song_state():
    if SONG_STATE["bpm"] is None:
        return {"loaded": False}

    return {
        "loaded": True,
        "bpm": SONG_STATE["bpm"],
        "chords": SONG_STATE["chords"]
    }


# ===============================
# Realtime chunk feedback
# ===============================

@app.post("/realtime/chunk")
async def realtime_chunk(
    audio: UploadFile = File(...),
    timestamp: float = Form(...)
):
    # 1️⃣ Find expected chord at timestamp
    expected = None
    for c in SONG_STATE["chords"]:
        if c["start"] <= timestamp <= c["end"]:
            expected = c
            break

    if expected is None or expected["chord"] == "N":
        return {
            "expected_chord": None,
            "confidence": 0.0,
            "is_correct": False
        }

    # 2️⃣ Save chunk
    chunk_path = "chunk.wav"
    with open(chunk_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    try:
        # 3️⃣ Extract chroma from user chunk
        y, sr = librosa.load(chunk_path, sr=44100, mono=True)

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        user_chroma = chroma.mean(axis=1)
        user_chroma /= (np.linalg.norm(user_chroma) + 1e-6)

        ref_chroma = np.array(expected["chroma"])

        # 4️⃣ Cosine similarity
        confidence = float(np.dot(user_chroma, ref_chroma))
        is_correct = confidence >= 0.7  # tunable threshold

        return {
            "expected_chord": expected["chord"],
            "confidence": round(confidence, 2),
            "is_correct": is_correct
        }

    finally:
        os.remove(chunk_path)
