# chord_engine.py

from madmom.audio.chroma import DeepChromaProcessor
from madmom.features.chords import DeepChromaChordRecognitionProcessor
from madmom.features.beats import RNNBeatProcessor
from madmom.features.tempo import TempoEstimationProcessor


def detect_bpm(audio_path: str) -> float:
    """
    Detect global BPM using beat activations
    """
    # Step 1: Beat activation
    beat_proc = RNNBeatProcessor()
    beat_activations = beat_proc(audio_path)

    # Step 2: Tempo estimation
    tempo_proc = TempoEstimationProcessor(fps=100)
    tempo = tempo_proc(beat_activations)

    bpm = round(float(tempo[0][0]), 2)
    return bpm


def analyze_audio(audio_path: str):
    """
    Analyze audio and return BPM + chord timeline
    """

    # -------- BPM --------
    bpm = detect_bpm(audio_path)

    # -------- CHORDS --------
    chroma = DeepChromaProcessor()(audio_path)
    chord_proc = DeepChromaChordRecognitionProcessor()
    chords = chord_proc(chroma)

    chord_list = [
        {
            "start": float(start),
            "end": float(end),
            "chord": label
        }
        for start, end, label in chords
    ]

    return {
        "bpm": bpm,
        "chords": chord_list
    }
