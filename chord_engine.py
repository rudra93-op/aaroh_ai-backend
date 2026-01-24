# chord_engine.py
import librosa
import numpy as np

from madmom.features.chords import (
    CNNChordFeatureProcessor,
    CRFChordRecognitionProcessor
)

# ===============================
# Helpers
# ===============================

def clean_time(t: float) -> float:
    return round(float(t), 2)


def normalize_bpm(bpm: float) -> float:
    if bpm <= 0:
        return 0.0
    while bpm > 160:
        bpm /= 2
    while bpm < 60:
        bpm *= 2
    return round(float(bpm), 2)


# ===============================
# BPM (FAST + STABLE)
# ===============================

def estimate_bpm(file_path: str) -> float:
    y, sr = librosa.load(
        file_path,
        sr=44100,
        mono=True,
        offset=10.0,      # skip intro
        duration=30.0     # limit for speed
    )

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempos = librosa.beat.tempo(
        onset_envelope=onset_env,
        sr=sr,
        aggregate=None
    )

    bpm = float(tempos.mean())
    return normalize_bpm(bpm)


# ===============================
# Chords + Reference Chroma
# ===============================

def extract_chords_with_chroma(file_path: str):
    # Chord recognition (madmom)
    feature_proc = CNNChordFeatureProcessor()
    chord_proc = CRFChordRecognitionProcessor()

    features = feature_proc(file_path)
    chords = chord_proc(features)

    # Load audio once for chroma
    y, sr = librosa.load(file_path, sr=44100, mono=True)

    result = []

    for start, end, label in chords:
        start_t = max(0.0, start)
        end_t = max(start_t + 0.05, end)

        # Slice audio for this chord
        s = int(start_t * sr)
        e = int(end_t * sr)
        segment = y[s:e]

        if len(segment) == 0:
            continue

        # Chroma extraction
        chroma = librosa.feature.chroma_cqt(y=segment, sr=sr)
        chroma_vec = chroma.mean(axis=1)

        # Normalize chroma
        chroma_vec /= (np.linalg.norm(chroma_vec) + 1e-6)

        result.append({
            "start": clean_time(start_t),
            "end": clean_time(end_t),
            "chord": label,
            "chroma": chroma_vec.tolist()
        })

    return result


# ===============================
# MAIN ANALYSIS
# ===============================

def analyze_audio(file_path: str):
    bpm = estimate_bpm(file_path)
    chords = extract_chords_with_chroma(file_path)

    return {
        "bpm": bpm,
        "chords": chords
    }
