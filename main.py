import streamlit as st
import os
import tempfile
import json
from dotenv import load_dotenv
from docx import Document
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from faster_whisper import WhisperModel
from langchain_core.prompts import PromptTemplate

# ============================================================
# 0) CONFIGURATION & INIT
# ============================================================

def init_app():
    load_dotenv()
    st.set_page_config(page_title="Technodysis Voice Chatbot", layout="centered")
    st.title("🎙️🤖 Technodysis Voice Chatbot")

def init_session_state():
    defaults = {
        "recon_stage": None,
        "final_query": "",
        "awaiting_correction_confirmation": False,
        "suggested_intent": "",
        "original_query": "",
        "top3_options": None,
        "awaiting_top3_choice": False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemma-3-27b-it",
        api_key=os.getenv("GOOGLE_API_KEY")
    )

# ============================================================
# 1) WHISPER MODEL (SPEECH-TO-TEXT)
# ============================================================

@st.cache_resource
def load_whisper_model():
    return WhisperModel("base", device="cpu", compute_type="int8")

def get_voice_query(whisper_model):
    audio_value = st.audio_input("Record your question", sample_rate=48000)

    if not audio_value:
        return ""

    st.audio(audio_value)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_value.getvalue())
        audio_path = f.name

    with st.spinner("Transcribing your voice..."):
        segments, info = whisper_model.transcribe(
            audio_path,
            language="en",
            task="transcribe"
        )
        voice_query = "".join(segment.text for segment in segments).strip()

    st.success("Transcription done ✅")
    st.write(f"**You said:** {voice_query}")
    return voice_query

# ============================================================
# 2) RAG SETUP (KNOWLEDGE BASE)
# ============================================================

def load_docx(path):
    if not os.path.exists(path):
        return "No document found."
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def chunk_text(text, size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

@st.cache_resource
def setup_rag():
    text = load_docx("Technodysis1.docx")
    chunks = chunk_text(text)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return chunks, embedder, index

def rag_search(query, chunks, embedder, index, top_k=3, threshold=0.4):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    scores, ids = index.search(q_emb, top_k)

    results = []
    for i, s in zip(ids[0], scores[0]):
        if s > threshold:
            results.append(chunks[i])
    return results

def answer_from_context(llm, query, context_chunks):
    if not context_chunks:
        return "Sorry, I don't have that information."
    
    context = "\n\n".join(context_chunks)
    prompt = f"""
    You are a company knowledge assistant.
    Answer clearly and completely using ONLY the context below.
    If the answer is not present, say: "Sorry, I don't have that information."

    Context:
    {context}

    Question:
    {query}

    Answer:
    """
    return llm.invoke(prompt).content

# ============================================================
# 3) SMART INTENT CLASSIFIER (Auto + Correction + Top 3 + Gibberish)
# ============================================================

def build_smart_intent_prompt():
    """
    Combined prompt: Handles Correction, Classification, Ambiguity, AND Gibberish.
    UPDATED: Strictly distinguishes between ACTION (doing a task) and INQUIRY (asking about it).
    ADDED: Dedicated 'greeting' intent to handle hi/hello smoothly.
    """
    template = """
    You are an intelligent intent classifier for a company bot.
    
    CRITICAL DISTINCTION:
    - If the user asks a QUESTION about a topic, the intent must be "convo".
    - Only select specific intents ("recon", "ocr", "kyc") if the user explicitly wants to PERFORM that action NOW.

    The valid intents are:
    1. recon: User wants to START reconciling files (e.g., "start recon", "compare these files").
    2. ocr: User wants to UPLOAD or EXTRACT text (e.g., "ocr this image", "extract text").
    3. kyc: User wants to VERIFY identity (e.g., "verify me", "do kyc", "upload passport").
    4. convo: Questions ABOUT the company/services (e.g., "Do you offer KYC?", "What is reconciliation?").
    5. greeting: Simple conversational greetings (e.g., "hi", "hello", "hey", "good morning").
    6. unknown: Gibberish or random characters.

    YOUR TASK:
    Analyze the user text and return a JSON object.

    SCENARIO A: CLEAR INTENT
    If the user clearly wants one specific thing, return:
    {{
        "type": "direct",
        "intent": "ocr",
        "confidence": 0.95
    }}

    SCENARIO B: TYPO / CORRECTION
    If the user makes a typo but the intent is obvious (e.g. "perform otr"), return:
    {{
        "type": "correction",
        "suggested_intent": "ocr",
        "original_term": "otr"
    }}

    SCENARIO C: AMBIGUOUS / VAGUE
    If the text is vague (e.g. "check this file"), return the top 3 likely intents:
    {{
        "type": "ambiguous",
        "options": [
            {{"intent": "ocr", "score": 0.45, "reason": "User mentioned 'document'"}},
            {{"intent": "recon", "score": 0.35, "reason": "Checking files implies comparison"}},
            {{"intent": "kyc", "score": 0.20, "reason": "Verification context"}}
        ]
    }}

    SCENARIO D: GIBBERISH / NONSENSE
    If the input makes NO sense, return:
    {{
        "type": "direct",
        "intent": "unknown",
        "confidence": 1.0
    }}

    EXAMPLES:
    User: "I want to perform reconn"
    JSON: {{"type": "correction", "suggested_intent": "recon", "original_term": "reconn"}}

    User: "Extract text from this invoice" (Action)
    JSON: {{"type": "direct", "intent": "ocr", "confidence": 0.98}}

    User: "Does your company do OCR?" (Inquiry)
    JSON: {{"type": "direct", "intent": "convo", "confidence": 0.99}}

    User: "Hello" (Greeting)
    JSON: {{"type": "direct", "intent": "greeting", "confidence": 1.00}}

    User: "What is the process for KYC?" (Inquiry)
    JSON: {{"type": "direct", "intent": "convo", "confidence": 0.99}}

    User: "asdf jkl lojz"
    JSON: {{"type": "direct", "intent": "unknown", "confidence": 1.0}}

    Now, analyze this text:
    User: "{query}"
    JSON Response:
    """
    return PromptTemplate(template=template, input_variables=["query"])

def analyze_intent_smart(llm, prompt_template, query):
    prompt = prompt_template.format(query=query)
    raw = llm.invoke(prompt).content.strip()
    
    # Clean up markdown formatting often returned by LLMs
    raw = raw.replace("```json", "").replace("```", "").strip()
    
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # If the LLM output is broken, assume it didn't understand
        return {"type": "direct", "intent": "unknown", "confidence": 0.0}

# ============================================================
# 4) UI HANDLERS & EXECUTION
# ============================================================

def get_query_from_ui(whisper_model):
    st.subheader("Choose Input Method")
    input_mode = st.radio("Mode:", ["⌨️ Type", "🎙️ Voice"], horizontal=True)

    if input_mode == "⌨️ Type":
        return st.text_input("Ask something").strip()
    
    return get_voice_query(whisper_model).strip()

def handle_final_intent(intent, llm, query, chunks, embedder, index):
    """Executes the final chosen intent."""
    
    # 1. Handle Unknown / Gibberish first
    if intent == "unknown":
        st.error("Sorry, I didn't understand what you mean. Can you please repeat?")
        return

    # 2. Handle Greetings instantly (Bypasses the RAG & Execution banner)
    if intent == "greeting":
        st.success("👋 Hello I am technodysis chatbot how can I help you")
        return

    # 3. Show Success Message for valid tasks/convo
    st.markdown(f"""
    <div style="background-color:#d4edda;padding:10px;border-radius:5px;border:1px solid #c3e6cb;">
        <h3 style="color:#155724;margin:0;">🚀 Executing: {intent.upper()}</h3>
    </div><br>
    """, unsafe_allow_html=True)

    # 4. Execute Logic
    if intent == "convo":
        results = rag_search(query, chunks, embedder, index)
        response = answer_from_context(llm, query, results)
        st.write(response)
        
    elif intent == "recon":
        st.session_state.recon_stage = "confirm"
        st.warning("Do you want to start Reconciliation? (yes / no)")
        
    elif intent == "ocr":
        st.info("📄 **OCR Module Active**: Please upload your image/PDF.")
        st.file_uploader("Upload Document", key="ocr_uploader")
        
    elif intent == "kyc":
        st.info("🪪 **KYC Module Active**: Ready for identity verification.")
        st.button("Start Verification Process")

# ============================================================
# 5) RECON FLOW LOGIC
# ============================================================

def run_recon_flow():
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

# ============================================================
# 6) MAIN APP LOGIC
# ============================================================

def main():
    init_app()
    init_session_state()

    llm = load_llm()
    whisper_model = load_whisper_model()
    chunks, embedder, index = setup_rag()
    
    intent_prompt = build_smart_intent_prompt()

    # --- 1. Get Input ---
    query = get_query_from_ui(whisper_model)
    
    # Only process if query exists and is new (or we are in a correction flow)
    if query and query != st.session_state.final_query:
        st.session_state.final_query = query
        
        # Reset specific states to avoid loops
        st.session_state.suggested_intent = ""
        st.session_state.top3_options = None
        st.session_state.awaiting_correction_confirmation = False
        st.session_state.awaiting_top3_choice = False
        st.session_state.recon_stage = None

        # --- 2. Analyze Intent ---
        result = analyze_intent_smart(llm, intent_prompt, query)
        
        # Route based on result type
        if result["type"] == "correction":
            st.session_state.suggested_intent = result["suggested_intent"]
            st.session_state.awaiting_correction_confirmation = True
            st.rerun()
            
        elif result["type"] == "ambiguous":
            st.session_state.top3_options = result["options"]
            st.session_state.awaiting_top3_choice = True
            st.rerun()
            
        else: # type == direct
            handle_final_intent(result["intent"], llm, query, chunks, embedder, index)

    # --- 3. Handle Pending Interactions ---
    
    # A) Correction Confirmation ("Did you mean...?")
    if st.session_state.awaiting_correction_confirmation:
        st.info(f"🧐 Did you mean **{st.session_state.suggested_intent.upper()}**?")
        col1, col2 = st.columns(2)
        
        if col1.button("✅ Yes"):
            intent = st.session_state.suggested_intent
            st.session_state.awaiting_correction_confirmation = False
            handle_final_intent(intent, llm, st.session_state.final_query, chunks, embedder, index)
            
        if col2.button("❌ No"):
            st.session_state.awaiting_correction_confirmation = False
            # Fallback to general conversation
            handle_final_intent("convo", llm, st.session_state.final_query, chunks, embedder, index)

    # B) Top 3 Choice ("I'm not sure...")
    if st.session_state.awaiting_top3_choice:
        st.warning("⚠️ I'm not fully sure. Please choose what you meant:")
        
        options_list = st.session_state.top3_options
        # Create labels like "OCR (45%)"
        radio_options = [f"{opt['intent'].upper()} ({int(opt['score']*100)}%)" for opt in options_list]
        
        choice = st.radio("Select intent:", radio_options)
        
        if st.button("Confirm Selection"):
            # Extract "OCR" from "OCR (45%)"
            selected_intent = choice.split(" ")[0].lower()
            st.session_state.awaiting_top3_choice = False
            handle_final_intent(selected_intent, llm, st.session_state.final_query, chunks, embedder, index)

    # --- 4. Run any active flows (like Recon uploads) ---
    run_recon_flow()

if __name__ == "__main__":
    main()