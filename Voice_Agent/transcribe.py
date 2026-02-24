import numpy as np
from faster_whisper import WhisperModel

def load_transcriber():
    """Load the Whisper model once when the server starts."""
    return WhisperModel("base", device="cpu", compute_type="int8")

def transcribe_audio(audio_bytes, model):
    """Takes raw audio bytes from frontend and returns English text + detected language."""
    # Convert raw bytes to the format Whisper expects
    audio_np = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
    
    # Transcribe and translate to English automatically
    segments, info = model.transcribe(audio_np, task="translate")
    english_text = "".join(segment.text for segment in segments).strip()
    
    return english_text, info.language