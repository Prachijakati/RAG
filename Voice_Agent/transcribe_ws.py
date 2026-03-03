# transcribe.py

from faster_whisper import WhisperModel
import time
import os

# 🔥 Load model once at import time
model = WhisperModel(
    "base",                  # Keep multilingual accuracy
    device="cpu",
    compute_type="int8",
    cpu_threads=os.cpu_count()
)


def transcribe_file(file_path: str) -> str:
    """
    Transcribes audio file to English text (auto-detect language + translate).
    Optimized for CPU performance without sacrificing model quality.
    """

    start_time = time.time()

    segments, info = model.transcribe(
        file_path,
        task="translate",              # Multilingual → English
        beam_size=1,                   # Greedy decoding (faster)
        best_of=1,
        temperature=0,
        condition_on_previous_text=False
    )

    transcript = "".join(
        [segment.text for segment in segments]
    ).strip()

    end_time = time.time()

    print(f"[STT] Detected Language: {info.language}")
    print(f"[STT] Processing Time: {end_time - start_time:.3f} sec")

    return transcript