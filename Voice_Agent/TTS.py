import asyncio
import edge_tts
import tempfile
import os

# Fetch available voices once when the server starts
try:
    AVAILABLE_VOICES = asyncio.run(edge_tts.list_voices())
except Exception as e:
    print(f"Failed to load voices: {e}")
    AVAILABLE_VOICES = []

def get_dynamic_voice(iso_code):
    """Dynamically finds a voice matching the requested language code."""
    for voice in AVAILABLE_VOICES:
        if voice['Locale'].lower().startswith(iso_code.lower()):
            return voice['ShortName']
    return "en-US-AriaNeural" # Fallback if no specific voice is found

async def _generate_audio_file(text, voice, path):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(path)

def text_to_audio(text, target_iso="en"):
    """Generates audio dynamically based on the target language."""
    voice = get_dynamic_voice(target_iso)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        output_path = tmp_file.name

    asyncio.run(_generate_audio_file(text, voice, output_path))
    
    with open(output_path, "rb") as f:
        audio_bytes = f.read()
        
    os.remove(output_path) 
    return audio_bytes