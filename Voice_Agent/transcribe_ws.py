# transcribe.py

from faster_whisper import WhisperModel

# Load once
model = WhisperModel("base", device="cpu", compute_type="int8")

def transcribe_file(file_path):
    segments, info = model.transcribe(file_path, task="translate")
    text = "".join([segment.text for segment in segments])
    return text.strip()