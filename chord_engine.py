import librosa
import numpy as np

from madmom.features.chords import (
    CNNChordFeatureProcessor,
    CRFChordRecognitionProcessor
)

# ---------------- helpers ----------------

def clean_time(t: float) -> float:
    """Remove floating garbage like 2.40000000000004"""
    return round(float(t), 2)


def normalize_bpm(bpm: float) -> float:
    """Fix half / double tempo errors"""
    if bpm <= 0:
        return 0.0

    while bpm > 160:
        bpm /= 2
    while bpm < 60:
        bpm *= 2

    return round(float(bpm), 2)


def estimate_bpm(file_path: str) -> float:
    """
    FAST + STABLE BPM
    - skip intro
    - limit duration
    """
    y, sr = librosa.load(
        file_path,
        sr=44100,
        mono=True,
        offset=10.0,      # skip intro
        duration=30.0     # speed
    )

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    tempos = librosa.beat.tempo(
        onset_envelope=onset_env,
        sr=sr,
        aggregate=None
    )

    bpm = float(np.mean(tempos))
    return normalize_bpm(bpm)


# ---------------- main ----------------

def analyze_audio(file_path: str):
    """
    OFFLINE SONG ANALYSIS
    Returns:
    - BPM
    - Chords with timestamps
    - Reference chroma per chord (for realtime comparison)
    """

    # 1️⃣ BPM (fast, stable)
    bpm = estimate_bpm(file_path)

    # 2️⃣ Load audio once (for chroma reference)
    y, sr = librosa.load(file_path, sr=44100, mono=True)

    hop_length = 512
    fps = sr / hop_length

    # Raw 12-bin chroma (THIS is for realtime comparison)
    chroma = librosa.feature.chroma_cqt(
        y=y,
        sr=sr,
        hop_length=hop_length
    )

    # 3️⃣ Chord recognition (madmom)
    feature_proc = CNNChordFeatureProcessor()
    chord_proc = CRFChordRecognitionProcessor()

    features = feature_proc(file_path)
    chords = chord_proc(features)

    chord_list = []

    for start, end, label in chords:
        start_f = int(start * fps)
        end_f = int(end * fps)

        if end_f <= start_f or end_f > chroma.shape[1]:
            continue

        # Reference chroma for this chord
        ref_chroma = chroma[:, start_f:end_f].mean(axis=1)
        ref_chroma = ref_chroma / (np.linalg.norm(ref_chroma) + 1e-6)

        chord_list.append({
            "start": clean_time(start),
            "end": clean_time(end),
            "chord": label,
            "chroma": np.round(ref_chroma, 4).tolist()
        })

    return {
        "bpm": bpm,
        "chords": chord_list
    }
