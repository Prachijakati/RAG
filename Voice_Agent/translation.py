from deep_translator import GoogleTranslator

def translate_text(text, target_iso="en"):
    """Translates English text into the target language."""
    if target_iso == "en":
        return text
        
    try:
        translator = GoogleTranslator(source='en', target=target_iso)
        return translator.translate(text)
    except Exception as e:
        print(f"Translation Error: {e}")
        return text # Fallback to English if it fails