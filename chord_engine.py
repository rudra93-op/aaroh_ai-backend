# chord_engine.py
import librosa
import numpy as np
from madmom.features.chords import (
    CNNChordFeatureProcessor,
    CRFChordRecognitionProcessor
)

# ---------------- helpers ----------------

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


def estimate_bpm(file_path: str) -> float:
    y, sr = librosa.load(
        file_path,
        sr=44100,
        mono=True,
        offset=10.0,   # skip intro
        duration=30.0  # speed
    )
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempos = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
    return normalize_bpm(float(tempos.mean()))


def extract_reference_chroma(y, sr, start, end):
    y_seg = y[int(start * sr): int(end * sr)]
    if len(y_seg) < sr * 0.3:
        return None

    chroma = librosa.feature.chroma_cqt(y=y_seg, sr=sr)
    ref = chroma.mean(axis=1)
    ref /= (np.linalg.norm(ref) + 1e-6)
    return ref.tolist()


# ---------------- main ----------------

def analyze_audio(file_path: str):
    # Load full audio once
    y, sr = librosa.load(file_path, sr=44100, mono=True)

    # BPM
    bpm = estimate_bpm(file_path)

    # Chord recognition
    feature_proc = CNNChordFeatureProcessor()
    chord_proc = CRFChordRecognitionProcessor()

    features = feature_proc(file_path)
    chords = chord_proc(features)

    chord_list = []
    for start, end, label in chords:
        chroma = extract_reference_chroma(y, sr, start, end)

        chord_list.append({
            "start": clean_time(start),
            "end": clean_time(end),
            "chord": label,
            "chroma": chroma
        })

    return {
        "bpm": bpm,
        "chords": chord_list
    }
