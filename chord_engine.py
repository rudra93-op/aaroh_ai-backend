import librosa
import os

from madmom.features.chords import (
    CNNChordFeatureProcessor,
    CRFChordRecognitionProcessor
)


# ---------- helpers ----------
def normalize_bpm(bpm: float) -> float:
    if bpm > 140:
        bpm = bpm / 2
    return round(float(bpm), 2)


def clean_time(t: float) -> float:
    return round(float(t), 2)


# ---------- main ----------
def analyze_audio(file_path: str):
    # 1️⃣ Load audio for BPM (librosa handles mp3/wav)
    y, sr = librosa.load(file_path, sr=44100, mono=True)

    # BPM only from first 30 sec (FAST)
    y_short = y[: sr * 30]
    bpm, _ = librosa.beat.beat_track(y=y_short, sr=sr)
    bpm = normalize_bpm(bpm)

    # 2️⃣ Chord features (CORRECT processor)
    feature_proc = CNNChordFeatureProcessor()
    chord_proc = CRFChordRecognitionProcessor()

    features = feature_proc(file_path)
    chords = chord_proc(features)

    chord_list = []
    for start, end, label in chords:
        chord_list.append({
            "start": clean_time(start),
            "end": clean_time(end),
            "chord": label
        })

    return {
        "bpm": bpm,
        "chords": chord_list
    }
