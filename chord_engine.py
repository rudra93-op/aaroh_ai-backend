from madmom.audio.chroma import DeepChromaProcessor
from madmom.features.chords import DeepChromaChordRecognitionProcessor

chroma_proc = DeepChromaProcessor()
chord_rec = DeepChromaChordRecognitionProcessor()

def predict_chords(wav_path):
    chroma = chroma_proc(wav_path)
    chords = chord_rec(chroma)

    result = []
    for start, end, chord in chords:
        result.append({
            "start": round(float(start), 2),
            "end": round(float(end), 2),
            "chord": chord
        })
    return result
