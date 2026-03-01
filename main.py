from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid  # ✅ Import UUID for unique filenames
import numpy as np
import librosa
import subprocess
import base64
import logging
import warnings
from chord_engine import analyze_audio

# ✅ 1. SETUP LOGGING & SILENCE WARNINGS (ताकि कंसोल क्लीन रहे)
logging.getLogger('librosa').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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
# Upload Song (HTTP)
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

# ==========================================
# ⚡ NEW: WEBSOCKET REALTIME ENDPOINT ⚡
# ==========================================
@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ Client connected to WebSocket")
    
    try:
        while True:
            # 1. Receive JSON payload from React
            data = await websocket.receive_json()
            
            beat_index = data.get("beatIndex")
            timestamp = data.get("timestamp", 0.0)
            audio_b64 = data.get("audio_data")

            if not audio_b64:
                continue

            # 2. Find Expected Chord
            expected = None
            for c in SONG_STATE["chords"]:
                if c["start"] <= timestamp <= c["end"]:
                    expected = c
                    break

            # If no chord is playing (N.C.)
            if not expected or expected["chord"] == "N":
                await websocket.send_json({
                    "beatIndex": beat_index,
                    "expected_chord": None,
                    "detected_chord": "N.C.",
                    "is_match": False,
                    "confidence": 0.0
                })
                continue

            # 3. Decode Audio from Base64 & Save Temp File
            audio_bytes = base64.b64decode(audio_b64)
            unique_id = uuid.uuid4().hex
            chunk_path = f"chunk_{unique_id}.webm"
            wav_path = f"chunk_{unique_id}.wav"

            try:
                with open(chunk_path, "wb") as f:
                    f.write(audio_bytes)

                # Convert to WAV (Fast Settings: 22050Hz)
                subprocess.run(
                    ["ffmpeg", "-y", "-i", chunk_path, "-ar", "22050", "-ac", "1", wav_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

                if not os.path.exists(wav_path):
                    raise Exception("FFmpeg conversion failed")

                # 4. Fast Analysis
                y, sr = librosa.load(wav_path, sr=22050, mono=True, duration=1.0)
                
                if len(y) == 0:
                    raise Exception("Empty audio file")

                chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
                user_chroma = chroma.mean(axis=1)
                
                norm = np.linalg.norm(user_chroma)
                if norm > 0: user_chroma /= norm

                ref_chroma = np.array(expected["chroma"])
                confidence = float(np.dot(user_chroma, ref_chroma))
                is_correct = confidence >= 0.7 

                # 5. Send Instant Reply to React
                await websocket.send_json({
                    "beatIndex": beat_index,
                    "expected_chord": expected["chord"],
                    "detected_chord": expected["chord"] if is_correct else "Unknown",
                    "is_match": is_correct,
                    "confidence": round(confidence, 3)
                })

            except Exception as e:
                print(f"⚠️ WS Analysis Error: {e}")
                await websocket.send_json({
                    "beatIndex": beat_index,
                    "expected_chord": expected["chord"],
                    "detected_chord": "Error",
                    "is_match": False,
                    "confidence": 0.0
                })
            finally:
                # Cleanup temp files instantly
                if os.path.exists(chunk_path): 
                    try: os.remove(chunk_path) 
                    except: pass
                if os.path.exists(wav_path): 
                    try: os.remove(wav_path) 
                    except: pass

    except WebSocketDisconnect:
        print("❌ Client disconnected from WebSocket")
    except Exception as e:
        print(f"⚠️ WebSocket General Error: {e}")


# ===============================
# Realtime Chunk Feedback (OLD HTTP - Kept as fallback)
# ===============================
@app.post("/realtime/chunk")
async def realtime_chunk(
    audio: UploadFile = File(...),
    timestamp: float = Form(...)
):
    expected = None
    for c in SONG_STATE["chords"]:
        if c["start"] <= timestamp <= c["end"]:
            expected = c
            break

    if not expected or expected["chord"] == "N":
        return {"expected_chord": None, "confidence": 0.0, "is_correct": False}

    unique_id = uuid.uuid4().hex
    chunk_path = f"chunk_{unique_id}.webm"
    wav_path = f"chunk_{unique_id}.wav"

    try:
        with open(chunk_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

        subprocess.run(
            ["ffmpeg", "-y", "-i", chunk_path, "-ar", "22050", "-ac", "1", wav_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        if not os.path.exists(wav_path):
             return {"expected_chord": expected["chord"], "confidence": 0.0, "is_correct": False}

        y, sr = librosa.load(wav_path, sr=22050, mono=True, duration=1.0)

        if len(y) == 0:
             return {"expected_chord": expected["chord"], "confidence": 0.0, "is_correct": False}

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        user_chroma = chroma.mean(axis=1)
        
        norm = np.linalg.norm(user_chroma)
        if norm > 0: user_chroma /= norm

        ref_chroma = np.array(expected["chroma"])
        confidence = float(np.dot(user_chroma, ref_chroma))
        is_correct = confidence >= 0.7 

        return {
            "expected_chord": expected["chord"],
            "confidence": round(confidence, 3),
            "is_correct": is_correct
        }

    except Exception as e:
        print(f"❌ Realtime HTTP Error: {e}")
        return { "expected_chord": expected["chord"] if expected else None, "confidence": 0.0, "is_correct": False }

    finally:
        if os.path.exists(chunk_path):
            try: os.remove(chunk_path)
            except: pass
        if os.path.exists(wav_path):
            try: os.remove(wav_path)
            except: pass