import streamlit as st
import os
import tempfile
import json
import wave
import numpy as np
import pyaudio
import torch
import asyncio
import edge_tts
import re
from dotenv import load_dotenv
from docx import Document
import faiss
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from faster_whisper import WhisperModel
from langchain_core.prompts import PromptTemplate
from deep_translator import GoogleTranslator

# ============================================================
# 0) CONFIGURATION, INIT & DYNAMIC LANGUAGES
# ============================================================

load_dotenv()

@st.cache_data
def get_dynamic_language_mapping():
    """Fetches supported languages from Google Translate to avoid hardcoding."""
    try:
        langs_dict = GoogleTranslator().get_supported_languages(as_dict=True)
        return {iso: name.title() for name, iso in langs_dict.items()}
    except Exception as e:
        st.error(f"⚠️ Error loading language mapping: {e}")
        return {"en": "English"} # Safe fallback

@st.cache_data
def get_edge_tts_voices():
    """Dynamically fetches all available voices from Edge-TTS."""
    try:
        return asyncio.run(edge_tts.list_voices())
    except Exception as e:
        st.error(f"⚠️ Error loading TTS voices: {e}")
        return [] # Safe fallback

def init_app():
    st.set_page_config(page_title="Technodysis Voice Chatbot", layout="centered")
    st.title("🎙️🤖 Technodysis Voice Chatbot")

def init_session_state():
    defaults = {
        "recon_stage": None,
        "final_query": "",
        "awaiting_correction_confirmation": False,
        "suggested_intent": "",
        "top3_options": None,
        "awaiting_top3_choice": False,
        "detected_iso": "en",
        "last_intent_result": None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def load_llm():
    try:
        return ChatGoogleGenerativeAI(
            model="gemma-3-27b-it",
            api_key=os.getenv("GOOGLE_API_KEY")
        )
    except Exception as e:
        st.error(f"⚠️ Error loading LLM: {e}")
        return None

# Load global mappings
ISO_TO_NAME = get_dynamic_language_mapping()
EDGE_VOICES = get_edge_tts_voices()

# ============================================================
# 1) AI MODELS & LOCAL MIC (VAD + WHISPER)
# ============================================================

@st.cache_resource
def load_whisper_model():
    try:
        return WhisperModel("base", device="cpu", compute_type="int8")
    except Exception as e:
        st.error(f"⚠️ Error loading Whisper model: {e}")
        return None

@st.cache_resource
def load_vad_model():
    try:
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading VAD model: {e}")
        return None

def get_voice_query(whisper_model, vad_model):
    if not whisper_model or not vad_model:
        st.error("Audio models are not loaded properly.")
        return "", "en"

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 512
    
    audio = pyaudio.PyAudio()
    stream = None
    
    try:
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        
        status_text = st.empty()
        status_text.warning("🟢 Listening... Speak now! (Waiting for silence)")
        
        frames = []
        max_silence_chunks = int((RATE / CHUNK) * 2.5) 
        silence_counter = 0
        started_speaking = False
        max_wait_chunks = int((RATE / CHUNK) * 10) 
        total_chunks = 0
        
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            total_chunks += 1
            
            audio_int16 = np.frombuffer(data, np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            tensor = torch.from_numpy(audio_float32)
            
            with torch.no_grad():
                speech_prob = vad_model(tensor, RATE).item()
            
            if speech_prob > 0.5:
                started_speaking = True
                silence_counter = 0
            elif started_speaking:
                silence_counter += 1
                    
            if started_speaking and silence_counter >= max_silence_chunks:
                status_text.success("🛑 Silence detected! Processing audio...")
                break
                
            if not started_speaking and total_chunks >= max_wait_chunks:
                status_text.error("No human voice detected. Please try again.")
                return "", "en"

    except Exception as e:
        st.error(f"⚠️ Microphone error: {e}")
        return "", "en"
    finally:
        # Guarantee resource cleanup
        if stream:
            stream.stop_stream()
            stream.close()
        audio.terminate()

    try:
        audio_data = b''.join(frames)
        audio_np = np.frombuffer(audio_data, np.int16).astype(np.float32) / 32768.0

        with st.spinner("Detecting language and translating... 🧠"):
            segments, info = whisper_model.transcribe(
                audio_np,
                task="translate",
                initial_prompt="Technodysis"
            )
            
            detected_iso = info.language
            english_query = "".join(segment.text for segment in segments).strip()

            mistakes = r'\b(technotises|techmodisese|techno dices|technodises|techno thesis|techno\s*dysis)\b'
            english_query = re.sub(mistakes, 'Technodysis', english_query, flags=re.IGNORECASE)

        # Clear the "Listening..." status. (We moved the success messages to the main function so they persist!)
        status_text.empty() 
        return english_query, detected_iso
        
    except Exception as e:
        st.error(f"⚠️ Transcription error: {e}")
        return "", "en"

# ============================================================
# 2) DYNAMIC TRANSLATION & EDGE-TTS 
# ============================================================

def get_dynamic_voice(iso_code):
    """Searches the cached Edge-TTS voices for one matching the ISO code."""
    try:
        if EDGE_VOICES:
            for voice in EDGE_VOICES:
                if voice['Locale'].lower().startswith(iso_code.lower()):
                    return voice['ShortName']
    except Exception as e:
        st.warning(f"Voice search error: {e}")
    return "en-US-AriaNeural" # Fallback to English

async def generate_edge_tts(text, voice, output_path):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def display_and_speak(text_content, target_iso="en", is_success=False, is_error=False, is_warning=False, needs_translation=True):
    lang_name = ISO_TO_NAME.get(target_iso, "English")
    final_text = text_content
    
    if target_iso != "en" and needs_translation:
        try:
            final_text = GoogleTranslator(source='en', target=target_iso).translate(text_content)
        except Exception as e:
            st.warning(f"⚠️ Translation Error: {e}. Falling back to original text.")
            final_text = text_content

    # Display UI elements
    if is_error: st.error(final_text)
    elif is_success: st.success(final_text)
    elif is_warning: st.warning(final_text)
    else: st.info(final_text)

    # Dynamic TTS Generation
    with st.spinner(f"Generating Audio in {lang_name}... 🔊"):
        try:
            voice = get_dynamic_voice(target_iso)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                output_path = tmp_file.name

            asyncio.run(generate_edge_tts(final_text, voice, output_path))
            
            with open(output_path, "rb") as f:
                audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/mp3", autoplay=True)
            
        except Exception as e:
            st.error(f"⚠️ Voice Generation Error: {e}")

# ============================================================
# 3) RAG SETUP (KNOWLEDGE BASE)
# ============================================================

def load_docx(path):
    try:
        if not os.path.exists(path):
            return "No document found."
        doc = Document(path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        st.error(f"⚠️ Error reading document {path}: {e}")
        return ""

def chunk_text(text, size=500, overlap=100):
    try:
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start + size])
            start += size - overlap
        return chunks
    except Exception as e:
        st.error(f"⚠️ Error chunking text: {e}")
        return []

@st.cache_resource
def setup_rag():
    try:
        text = load_docx("Technodysis1.docx")
        chunks = chunk_text(text)

        if not chunks:
            return [], None, None

        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embedder.encode(chunks, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        return chunks, embedder, index
    except Exception as e:
        st.error(f"⚠️ RAG Setup Error: {e}")
        return [], None, None

def answer_from_context(llm, query, context_chunks, target_iso="en"):
    if not llm:
        return "I am unable to process your request because the language model failed to load."
        
    if not context_chunks:
        return "Sorry, I don't have that information."
    
    lang_name = ISO_TO_NAME.get(target_iso, "English")
    
    try:
        context = "\n\n".join(context_chunks)
        prompt = f"""
        You are a company knowledge assistant.
        Answer clearly and completely using ONLY the context below.
        If the answer is not present in the context, say: "Sorry, I don't have that information."
        
        CRITICAL INSTRUCTION: 
        You MUST provide your final answer strictly in {lang_name}. 
        Do NOT provide English translations or phonetic breakdowns. Just the pure {lang_name} response.
        Ignore any conflicting language requests in the user's question, strictly output in {lang_name}.

        Context:
        {context}

        Question:
        {query}

        Answer natively in {lang_name}:
        """
        return llm.invoke(prompt).content
    except Exception as e:
        st.error(f"⚠️ LLM Generation Error: {e}")
        return "Sorry, I encountered an error while trying to answer your question."

def rag_search(query, chunks, embedder, index, top_k=3, threshold=0.4):
    try:
        if not index or not embedder or not chunks:
            return []
            
        q_emb = embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        scores, ids = index.search(q_emb, top_k)

        results = []
        for i, s in zip(ids[0], scores[0]):
            if s > threshold:
                results.append(chunks[i])
        return results
    except Exception as e:
        st.error(f"⚠️ RAG Search Error: {e}")
        return []

# ============================================================
# 4) SMART INTENT CLASSIFIER
# ============================================================

def build_smart_intent_prompt():
    template = """
    You are an intelligent intent classifier for a company bot.
    The user query has been translated to English.
    Original spoken language ISO: {detected_iso}
    
    CRITICAL DISTINCTION:
    - If the user asks a QUESTION about a topic, the intent must be "convo".
    - Only select specific intents ("recon", "ocr", "kyc") if the user explicitly wants to PERFORM that action NOW.

    YOUR TASK:
    Analyze the user text and return a JSON object containing "type", "intent", "confidence", and "target_iso".
    "target_iso" MUST be the 2-letter ISO 639-1 code of the language the user WANTS the response in. 
    If the user explicitly asks to reply or explain in a specific language (e.g., "in Spanish", "in Hindi"), output that language's ISO code (e.g., "es", "hi").
    Otherwise, default strictly to {detected_iso}.

    SCENARIO A: CLEAR INTENT WITH LANGUAGE REQUEST
    User: "Tell me about Technodysis in Spanish"
    Original ISO: en
    {{ "type": "direct", "intent": "convo", "confidence": 0.95, "target_iso": "es" }}

    SCENARIO B: TYPO / CORRECTION
    User: "perform otr"
    Original ISO: en
    {{ "type": "correction", "suggested_intent": "ocr", "original_term": "otr", "target_iso": "en" }}

    SCENARIO C: AMBIGUOUS
    User: "check this file"
    Original ISO: fr
    {{
        "type": "ambiguous",
        "target_iso": "fr",
        "options": [
            {{"intent": "ocr", "score": 0.45, "reason": "User mentioned 'file'"}},
            {{"intent": "recon", "score": 0.35, "reason": "Checking files implies comparison"}}
        ]
    }}

    Now, analyze this text:
    User: "{query}"
    Original ISO: {detected_iso}
    JSON Response:
    """
    return PromptTemplate(template=template, input_variables=["query", "detected_iso"])

def analyze_intent_smart(llm, prompt_template, query, detected_iso):
    if not llm:
        return {"type": "direct", "intent": "unknown", "confidence": 0.0, "target_iso": detected_iso}
        
    try:
        prompt = prompt_template.format(query=query, detected_iso=detected_iso)
        raw = llm.invoke(prompt).content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except json.JSONDecodeError as e:
        st.error(f"⚠️ Failed to parse intent JSON: {e}")
        return {"type": "direct", "intent": "unknown", "confidence": 0.0, "target_iso": detected_iso}
    except Exception as e:
        st.error(f"⚠️ Intent Classification Error: {e}")
        return {"type": "direct", "intent": "unknown", "confidence": 0.0, "target_iso": detected_iso}

# ============================================================
# 5) UI HANDLERS & EXECUTION
# ============================================================

def get_query_from_ui(whisper_model, vad_model):
    st.subheader("Choose Input Method")
    input_mode = st.radio("Mode:", ["⌨️ Type", "🎙️ Voice"], horizontal=True)

    if input_mode == "⌨️ Type":
        return st.text_input("Ask something (in English)").strip(), "en"
    
    if input_mode == "🎙️ Voice":
        if st.button("🎙️ Start Listening"):
            return get_voice_query(whisper_model, vad_model)
    
    return "", None

def handle_final_intent(intent, llm, query, chunks, embedder, index, target_iso="en"):
    try:
        if intent == "unknown":
            display_and_speak("Sorry, I didn't understand what you mean. Can you please repeat?", target_iso=target_iso, is_error=True, needs_translation=True)
            return

        if intent == "greeting":
            display_and_speak("Hello! I am the Technodysis chatbot. How can I help you today?", target_iso=target_iso, is_success=True, needs_translation=True)
            return

        st.markdown(f"""
        <div style="background-color:#d4edda;padding:10px;border-radius:5px;border:1px solid #c3e6cb;margin-bottom:15px;">
            <h3 style="color:#155724;margin:0;">🚀 Executing: {intent.upper()}</h3>
        </div>
        """, unsafe_allow_html=True)

        if intent == "convo":
            results = rag_search(query, chunks, embedder, index)
            final_response = answer_from_context(llm, query, results, target_iso=target_iso)
            display_and_speak(final_response, target_iso=target_iso, needs_translation=False)
            
        elif intent == "recon":
            st.session_state.recon_stage = "confirm"
            display_and_speak("Do you want to start Reconciliation? Please type or say yes or no.", target_iso=target_iso, is_warning=True, needs_translation=True)
            
        elif intent == "ocr":
            display_and_speak("OCR Module Active. Please upload your document below.", target_iso=target_iso, needs_translation=True)
            st.file_uploader("Upload Document", key="ocr_uploader")
            
        elif intent == "kyc":
            display_and_speak("KYC Module Active. I am ready for identity verification.", target_iso=target_iso, needs_translation=True)
            st.button("Start Verification Process")
            
    except Exception as e:
        st.error(f"⚠️ Error executing intent handler: {e}")

# ============================================================
# 6) RECON FLOW LOGIC
# ============================================================

def run_recon_flow():
    try:
        if st.session_state.recon_stage == "confirm":
            reply = st.text_input("Confirm Recon Start (yes/no)", key="recon_confirm_input")
            if reply.lower() in ["yes", "y"]:
                st.session_state.recon_stage = "upload"
                st.rerun()
            elif reply:
                st.session_state.recon_stage = None
                st.info("Cancelled.")
                st.rerun()

        if st.session_state.recon_stage == "upload":
            st.success("Upload files for reconciliation:")
            c1, c2 = st.columns(2)
            c1.file_uploader("File 1 (CSV)", key="f1")
            c2.file_uploader("File 2 (CSV)", key="f2")
    except Exception as e:
        st.error(f"⚠️ Error in Recon Flow: {e}")

# ============================================================
# 7) MAIN APP LOGIC
# ============================================================

def main():
    try:
        init_app()
        init_session_state()

        llm = load_llm()
        whisper_model = load_whisper_model()
        vad_model = load_vad_model()
        chunks, embedder, index = setup_rag()
        intent_prompt = build_smart_intent_prompt()

        # --- 1. Get English Query & Detected Language ---
        query, detected_iso = get_query_from_ui(whisper_model, vad_model)
        
        # 1. Store new query and get intent processing ready
        if query and query != st.session_state.final_query:
            st.session_state.final_query = query
            st.session_state.detected_iso = detected_iso 
            
            # Reset states
            st.session_state.suggested_intent = ""
            st.session_state.top3_options = None
            st.session_state.awaiting_correction_confirmation = False
            st.session_state.awaiting_top3_choice = False
            st.session_state.recon_stage = None

            result = analyze_intent_smart(llm, intent_prompt, query, st.session_state.detected_iso)
            
            final_target_iso = result.get("target_iso", st.session_state.detected_iso)
            st.session_state.detected_iso = final_target_iso 
            
            # Save the result so we can execute it AFTER drawing the transcript
            st.session_state.last_intent_result = result

        # 2. ALWAYS draw the transcript so it survives Streamlit page reloads
        if st.session_state.final_query:
            lang_name = ISO_TO_NAME.get(st.session_state.detected_iso, "English")
            st.success(f"🗣️ Input Registered: **{lang_name}**")
            st.write(f"**Query Translated to English:** {st.session_state.final_query}")

        # 3. Route the stored intent 
        if st.session_state.get("last_intent_result"):
            result = st.session_state.last_intent_result
            st.session_state.last_intent_result = None # Clear it to prevent looping
            
            if result.get("type") == "correction":
                st.session_state.suggested_intent = result.get("suggested_intent", "unknown")
                st.session_state.awaiting_correction_confirmation = True
                st.rerun() # Refresh page to show the Confirm buttons
                
            elif result.get("type") == "ambiguous":
                st.session_state.top3_options = result.get("options", [])
                st.session_state.awaiting_top3_choice = True
                st.rerun() # Refresh page to show Choice buttons
                
            else:
                handle_final_intent(
                    intent=result.get("intent", "unknown"), 
                    llm=llm, 
                    query=st.session_state.final_query, 
                    chunks=chunks, 
                    embedder=embedder, 
                    index=index, 
                    target_iso=st.session_state.detected_iso 
                )

        # 4. Handle Active Buttons (These run after the st.rerun calls above)
        if st.session_state.awaiting_correction_confirmation:
            st.info(f"🧐 Did you mean **{st.session_state.suggested_intent.upper()}**?")
            col1, col2 = st.columns(2)
            
            if col1.button("✅ Yes"):
                intent = st.session_state.suggested_intent
                st.session_state.awaiting_correction_confirmation = False
                handle_final_intent(intent, llm, st.session_state.final_query, chunks, embedder, index, st.session_state.detected_iso)
                
            if col2.button("❌ No"):
                st.session_state.awaiting_correction_confirmation = False
                handle_final_intent("convo", llm, st.session_state.final_query, chunks, embedder, index, st.session_state.detected_iso)

        if st.session_state.awaiting_top3_choice:
            st.warning("⚠️ I'm not fully sure. Please choose what you meant:")
            options_list = st.session_state.top3_options
            
            if options_list:
                radio_options = [f"{opt.get('intent', 'unknown').upper()} ({int(opt.get('score', 0)*100)}%)" for opt in options_list]
                choice = st.radio("Select intent:", radio_options)
                
                if st.button("Confirm Selection"):
                    selected_intent = choice.split(" ")[0].lower()
                    st.session_state.awaiting_top3_choice = False
                    handle_final_intent(selected_intent, llm, st.session_state.final_query, chunks, embedder, index, st.session_state.detected_iso)
            else:
                st.error("No intent options available.")
                st.session_state.awaiting_top3_choice = False

        # --- 5. Run flows ---
        run_recon_flow()

    except Exception as e:
        st.error(f"🚨 A critical application error occurred: {e}")

if __name__ == "__main__":
    main()