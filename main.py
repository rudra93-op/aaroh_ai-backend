from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid  # ✅ Import UUID for unique filenames
import numpy as np
import librosa
import subprocess
from chord_engine import analyze_audio

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# In-memory state
# ===============================
SONG_STATE = {
    "bpm": None,
    "chords": []
}

# ===============================
# Health Check
# ===============================
@app.get("/health")
@app.head("/health")
def health():
    return {"status": "ok"}

# ===============================
# Upload Song
# ===============================
@app.post("/upload")
async def upload_song(file: UploadFile = File(...)):
    # Generate unique filename
    unique_id = uuid.uuid4().hex
    ext = os.path.splitext(file.filename)[1] or ".mp3"
    temp_path = f"temp_{unique_id}{ext}"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run Analysis
        result = analyze_audio(temp_path)

        # Update State
        SONG_STATE["bpm"] = result["bpm"]
        SONG_STATE["chords"] = result["chords"]

        print(f"✅ Analyzed: {result['bpm']} BPM, {len(result['chords'])} Chords")

        return {
            "status": "song_loaded",
            "bpm": result["bpm"],
            "chord_count": len(result["chords"]),
            "chords": result["chords"]
        }

    except Exception as e:
        print(f"❌ Upload Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ===============================
# Get Song State
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
# Realtime Chunk Feedback (FIXED)
# ===============================
@app.post("/realtime/chunk")
async def realtime_chunk(
    audio: UploadFile = File(...),
    timestamp: float = Form(...)
):
    # 1️⃣ Find expected chord
    expected = None
    for c in SONG_STATE["chords"]:
        if c["start"] <= timestamp <= c["end"]:
            expected = c
            break

    if expected is None or expected["chord"] == "N":
        return {"expected_chord": None, "confidence": 0.0, "is_correct": False}

    # 2️⃣ Save chunk with CORRECT EXTENSION & UNIQUE NAME
    unique_id = uuid.uuid4().hex
    ext = os.path.splitext(audio.filename)[1] or ".webm"
    chunk_path = f"chunk_{unique_id}{ext}"
    wav_path = f"chunk_{unique_id}.wav"  # Always safe

    try:
        with open(chunk_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

        # Convert to WAV
        subprocess.run(
            ["ffmpeg", "-y", "-i", chunk_path, "-ac", "1", "-ar", "44100", wav_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

       y, sr = librosa.load(wav_path, sr=22050, mono=True, duration=1.0)

        if len(y) == 0:
            return {"expected_chord": expected["chord"], "confidence": 0.0, "is_correct": False}

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        user_chroma = chroma.mean(axis=1)

        norm = np.linalg.norm(user_chroma)
        if norm > 0:
            user_chroma /= norm

        ref_chroma = np.array(expected["chroma"])
        confidence = float(np.dot(user_chroma, ref_chroma))
        is_correct = confidence >= 0.7

        return {"expected_chord": expected["chord"], "confidence": round(confidence, 3), "is_correct": is_correct}

    except Exception as e:
        print(f"❌ Realtime Error: {e}")
        return {"expected_chord": expected["chord"] if expected else None, "confidence": 0.0, "is_correct": False}

    finally:
        if os.path.exists(chunk_path):
            os.remove(chunk_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)
