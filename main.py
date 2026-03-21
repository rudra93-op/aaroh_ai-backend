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
import time  # ✅ Import time to measure AI speed
from chord_engine import analyze_audio

# ✅ 1. SETUP LOGGING & SILENCE WARNINGS
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
# ⚡ NEW: WEBSOCKET REALTIME ENDPOINT (Fast & Accurate + Silence Detection) ⚡
# ==========================================
@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ [WS] Client connected successfully!")
    
    try:
        while True:
            # 1. Receive JSON payload from React
            data = await websocket.receive_json()
            
            beat_index = data.get("beatIndex")
            timestamp = data.get("timestamp", 0.0)
            audio_b64 = data.get("audio_data")

            if not audio_b64:
                print("⚠️ [WS] Received empty audio chunk")
                continue

            print(f"📥 [WS] Received chunk - Beat: {beat_index}, Time: {timestamp:.2f}s")
            start_time = time.time()  # ⏱️ Start timer

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

                # 4. Fast Analysis (Duration set to 0.5 for speed)
                y, sr = librosa.load(wav_path, sr=22050, mono=True, duration=0.5)
                
                if len(y) == 0:
                    raise Exception("Empty audio file")

                # ✅ SILENCE DETECTOR ADDED HERE
                # Measure the RMS energy (volume level) of the audio chunk
                rms = librosa.feature.rms(y=y)
                mean_rms = float(np.mean(rms))
                
                # Agar volume 0.01 se kam hai, matlab user kuch nahi baja raha (silence/background noise hai)
                if mean_rms < 0.01:
                    print(f"🔇 [AI] Silence Detected (Volume: {mean_rms:.4f}). Ignoring.")
                    await websocket.send_json({
                        "beatIndex": beat_index,
                        "expected_chord": expected["chord"],
                        "detected_chord": "Silence",
                        "is_match": False,
                        "confidence": 0.0
                    })
                    continue

                # ✅ Normal Analysis (if audio is loud enough)
                chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=512)
                user_chroma = chroma.mean(axis=1)
                
                norm = np.linalg.norm(user_chroma)
                if norm > 0: user_chroma /= norm

                ref_chroma = np.array(expected["chroma"])
                confidence = float(np.dot(user_chroma, ref_chroma))
                
                # Threshold adjusted for real human playing (0.65)
                is_correct = confidence >= 0.65 

                process_time = time.time() - start_time  # ⏱️ End timer

                print(f"✅ [AI Done] Processed in {process_time:.3f}s | Vol: {mean_rms:.3f} | Expected: {expected['chord']} | Conf: {confidence:.2f} | Match: {is_correct}")

                # 5. Send Instant Reply to React
                await websocket.send_json({
                    "beatIndex": beat_index,
                    "expected_chord": expected["chord"],
                    "detected_chord": expected["chord"] if is_correct else "Unknown",
                    "is_match": is_correct,
                    "confidence": round(confidence, 3)
                })

            except Exception as e:
                print(f"⚠️ [WS Error] Analysis failed: {e}")
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
        print("❌ [WS] Client disconnected")
    except Exception as e:
        print(f"⚠️ [WS] General Error: {e}")

# ===============================
# Realtime Chunk Feedback (OLD HTTP - Kept as fallback, updated for speed)
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

        # ✅ Speed update here as well
        y, sr = librosa.load(wav_path, sr=22050, mono=True, duration=0.5)

        if len(y) == 0:
             return {"expected_chord": expected["chord"], "confidence": 0.0, "is_correct": False}

        # ✅ STFT update here as well
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=512)
        user_chroma = chroma.mean(axis=1)
        
        norm = np.linalg.norm(user_chroma)
        if norm > 0: user_chroma /= norm

        ref_chroma = np.array(expected["chroma"])
        confidence = float(np.dot(user_chroma, ref_chroma))
        is_correct = confidence >= 0.65 

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