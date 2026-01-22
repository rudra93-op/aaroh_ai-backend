from fastapi import FastAPI, UploadFile, File, Form
import numpy as np
import librosa
import os
import shutil

from chord_engine import analyze_audio

app = FastAPI()

# ===============================
# In-memory song state
# (later replace with Redis / DB)
# ===============================
SONG_STATE = {
    "bpm": None,
    "chords": []
}

# ===============================
# HEALTH CHECK (Render / Uptime)
# ===============================
@app.get("/health")
@app.head("/health")
def health():
    return {"status": "ok"}


# ===============================
# UPLOAD SONG (offline analysis)
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
            "chord_count": len(result["chords"])
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ===============================
# REALTIME CHUNK FEEDBACK
# ===============================
@app.post("/realtime/chunk")
async def realtime_chunk(
    audio: UploadFile = File(...),
    timestamp: float = Form(...)
):
    """
    Realtime feedback API
    """

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

    # 2️⃣ Save chunk temporarily
    chunk_path = "chunk.wav"
    with open(chunk_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    try:
        # 3️⃣ Extract chroma from user chunk
        y, sr = librosa.load(chunk_path, sr=44100, mono=True)

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        user_chroma = chroma.mean(axis=1)
        user_chroma /= (np.linalg.norm(user_chroma) + 1e-6)

        # expected["chroma"] must be saved during upload (next step)
        ref_chroma = np.array(expected["chroma"])

        # 4️⃣ Cosine similarity
        confidence = float(np.dot(user_chroma, ref_chroma))
        is_correct = confidence >= 0.7  # threshold

        return {
            "expected_chord": expected["chord"],
            "confidence": round(confidence, 2),
            "is_correct": is_correct
        }

    finally:
        if os.path.exists(chunk_path):
            os.remove(chunk_path)
