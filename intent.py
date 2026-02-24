import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

def load_intent_model():
    """Load the LLM once when the server starts."""
    return ChatGoogleGenerativeAI(
        model="gemma-3-27b-it", 
        api_key=os.getenv("GOOGLE_API_KEY")
    )

def detect_target_language(llm, english_text, detected_iso):
    """
    The Detective: Reads the English text to see if the user explicitly 
    asked for a specific language (like 'in Kannada').
    """
    template = """
    The user's original spoken language was: {detected_iso}
    The user said (translated to English): "{query}"
    
    Did the user explicitly ask to reply or explain in a specific language (e.g., "in Kannada", "in Spanish")?
    If YES, output ONLY a JSON object with the 2-letter ISO code of that requested language. 
    If NO, output ONLY a JSON object with the original {detected_iso} code.
    
    Example outputs:
    {{"target_iso": "kn"}}
    {{"target_iso": "es"}}
    
    JSON Response:
    """
    prompt = PromptTemplate(template=template, input_variables=["query", "detected_iso"])
    
    try:
        # Ask Gemini to figure it out
        raw_response = llm.invoke(prompt.format(query=english_text, detected_iso=detected_iso)).content.strip()
        # Clean up the response
        raw_response = raw_response.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw_response)
        
        return result.get("target_iso", detected_iso)
        
    except Exception as e:
        print(f"Intent Error: {e}")
        return detected_iso # If it fails, fallback to what they originally spoke