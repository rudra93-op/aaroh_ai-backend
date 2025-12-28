import librosa
from madmom.features.chords import (
    CNNChordFeatureProcessor,
    CRFChordRecognitionProcessor
)

# ---------------- helpers ----------------

def normalize_bpm(bpm: float) -> float:
    """
    Fix half / double tempo errors
    """
    if bpm <= 0:
        return 0.0

    while bpm > 160:
        bpm /= 2
    while bpm < 60:
        bpm *= 2

    return round(float(bpm), 2)


def clean_time(t: float) -> float:
    """
    Remove floating garbage like 2.400000000000004
    """
    return round(float(t), 2)


def estimate_bpm(file_path: str) -> float:
    """
    FAST + STABLE BPM
    - skips intro
    - uses onset envelope
    - max 30s audio
    """
    y, sr = librosa.load(
        file_path,
        sr=44100,
        mono=True,
        offset=10.0,      # skip intro (VERY IMPORTANT)
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


# ---------------- main ----------------

def analyze_audio(file_path: str):
    # 1️⃣ BPM (FAST, SAFE)
    bpm = estimate_bpm(file_path)

    # 2️⃣ Chords (FULL SONG)
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
