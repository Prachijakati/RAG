from transcribe import load_transcriber, transcribe_audio
from intent import load_intent_model, detect_target_language
from translation import translate_text
from TTS import text_to_audio

# Load models on server start
whisper_model = load_transcriber()
llm_model = load_intent_model()

def process_user_audio_input(frontend_audio_bytes):
    """
    PHASE 1: Audio to LLM Input.
    Pass the raw audio bytes here. It returns the translated English text 
    and the requested target language so your LLM can do its job.
    """
    english_text, spoken_iso = transcribe_audio(frontend_audio_bytes, whisper_model)
    target_iso = detect_target_language(llm_model, english_text, spoken_iso)
    
    return english_text, target_iso

def process_bot_audio_output(llm_english_response, target_iso):
    """
    PHASE 2: LLM Output to Audio.
    Pass your LLM's final English response and the target_iso here.
    It returns the translated text and the audio bytes for the frontend.
    """
    translated_text = translate_text(llm_english_response, target_iso)
    audio_output_bytes = text_to_audio(translated_text, target_iso)
    
    return translated_text, audio_output_bytes