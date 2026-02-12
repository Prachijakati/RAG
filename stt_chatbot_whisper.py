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

from langchain.prompts import PromptTemplate, FewShotPromptTemplate


# ============================================================
# 0) CONFIG
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
        "original_query": ""
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
# 1) WHISPER MODEL
# ============================================================

@st.cache_resource
def load_whisper_model():
    return WhisperModel(
        "base",
        device="cpu",  # change to "cuda" if GPU available
        compute_type="int8"
    )


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
    st.write("**You said:**")
    st.write(voice_query)

    return voice_query


# ============================================================
# 2) RAG SETUP
# ============================================================

def load_docx(path):
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
If the answer is not present, say:
"Sorry, I don't have that information."

Context:
{context}

Question:
{query}

Answer:
"""
    return llm.invoke(prompt).content


# ============================================================
# 3) INTENT + CORRECTION PROMPTS
# ============================================================

def build_correction_prompt(llm):
    correction_examples = [
        {"text": "I want to perform otr on this image",
         "json": '{{"needs_confirmation": true, "suggested_intent": "ocr", "original_term": "otr"}}'},

        {"text": "can you do opr on this document",
         "json": '{{"needs_confirmation": true, "suggested_intent": "ocr", "original_term": "opr"}}'},

        {"text": "help me with kvc verification",
         "json": '{{"needs_confirmation": true, "suggested_intent": "kyc", "original_term": "kvc"}}'},

        {"text": "I want nyc verification for passport",
         "json": '{{"needs_confirmation": true, "suggested_intent": "kyc", "original_term": "nyc"}}'},

        {"text": "can you do reconcilliation of these files",
         "json": '{{"needs_confirmation": true, "suggested_intent": "recon", "original_term": "reconcilliation"}}'},

        {"text": "do rekkon for these two spreadsheets",
         "json": '{{"needs_confirmation": true, "suggested_intent": "recon", "original_term": "rekkon"}}'},

        {"text": "perform ocr on this image",
         "json": '{{"needs_confirmation": false, "suggested_intent": "ocr", "original_term": ""}}'},

        {"text": "I want to do kyc verification using aadhar",
         "json": '{{"needs_confirmation": false, "suggested_intent": "kyc", "original_term": ""}}'},

        {"text": "reconcile these two excel files",
         "json": '{{"needs_confirmation": false, "suggested_intent": "recon", "original_term": ""}}'},

        {"text": "what services does Technodysis provide",
         "json": '{{"needs_confirmation": false, "suggested_intent": "convo", "original_term": ""}}'},

        {"text": "tell me about your company",
         "json": '{{"needs_confirmation": false, "suggested_intent": "convo", "original_term": ""}}'},

        {"text": "tell me a joke",
         "json": '{{"needs_confirmation": false, "suggested_intent": "convo", "original_term": ""}}'},
    ]

    example_prompt = PromptTemplate(
        input_variables=["text", "json"],
        template="Text: {text}\nResponse: {json}\n"
    )

    return FewShotPromptTemplate(
        examples=correction_examples,
        example_prompt=example_prompt,
        prefix=(
            "You are an intent correction assistant.\n\n"
            "Your job is ONLY to detect if the user likely misspelled or mispronounced ONE of these intents:\n"
            "1) recon (reconciliation / compare two files)\n"
            "2) ocr (extract text from image/document)\n"
            "3) kyc (identity verification like aadhar/passport)\n\n"
            "STRICT RULES:\n"
            "- If user clearly means recon/ocr/kyc but wrote a wrong word -> needs_confirmation=true\n"
            "- If user clearly wrote recon/ocr/kyc correctly -> needs_confirmation=false\n"
            "- If the text is general conversation OR unrelated -> needs_confirmation=false AND suggested_intent='convo'\n"
            "- NEVER guess kyc/ocr/recon for normal questions.\n"
            "- Output MUST be valid JSON ONLY. No explanation.\n\n"
            "Examples:\n"
        ),
        suffix="Now analyze:\nText: {text}\nResponse:",
        input_variables=["text"]
    )


def detect_possible_correction(llm, correction_prompt, text: str) -> dict:
    prompt = correction_prompt.format(text=text)
    raw = llm.invoke(prompt).content.strip()

    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(raw)

        needs_confirmation = bool(result.get("needs_confirmation", False))
        suggested_intent = result.get("suggested_intent", "convo")
        original_term = result.get("original_term", "")

        if suggested_intent not in ["convo", "recon", "ocr", "kyc"]:
            return {"needs_confirmation": False, "suggested_intent": "convo", "original_term": ""}

        return {
            "needs_confirmation": needs_confirmation,
            "suggested_intent": suggested_intent,
            "original_term": original_term
        }

    except:
        return {"needs_confirmation": False, "suggested_intent": "convo", "original_term": ""}


def build_intent_prompt():
    intent_examples = [
        {"text": "What services does Technodysis provide?", "intent": "convo"},
        {"text": "Tell me about your company", "intent": "convo"},
        {"text": "I need to reconcile two files", "intent": "recon"},
        {"text": "Compare these two spreadsheets", "intent": "recon"},
        {"text": "Can you perform OCR on this document?", "intent": "ocr"},
        {"text": "Extract text from image", "intent": "ocr"},
        {"text": "Help me with KYC verification", "intent": "kyc"},
        {"text": "Verify customer identity", "intent": "kyc"},
    ]

    example_prompt = PromptTemplate(
        input_variables=["text", "intent"],
        template="Text: {text}\nIntent: {intent}\n"
    )

    return FewShotPromptTemplate(
        examples=intent_examples,
        example_prompt=example_prompt,
        prefix=(
            "Classify the user intent into ONLY ONE word:\n"
            "convo, recon, ocr, kyc, unknown\n\n"
            "Examples:\n"
        ),
        suffix="Now classify:\nText: {text}\nIntent:",
        input_variables=["text"]
    )


def classify_intent(llm, intent_prompt, text: str) -> str:
    prompt = intent_prompt.format(text=text)
    intent = llm.invoke(prompt).content.strip().lower()
    return intent if intent in ["convo", "recon", "ocr", "kyc"] else "unknown"


# ============================================================
# 4) HELPERS
# ============================================================

def is_greeting(q):
    return q.lower().strip() in ["hi", "hello", "hey"]


def get_query_from_ui(whisper_model):
    st.subheader("Choose Input Method")

    input_mode = st.radio(
        "How do you want to ask?",
        ["⌨️ Type", "🎙️ Voice"],
        horizontal=True
    )

    if input_mode == "⌨️ Type":
        return st.text_input("Ask something").strip()

    return get_voice_query(whisper_model).strip()


# ============================================================
# 5) CORRECTION CONFIRMATION UI
# ============================================================

def handle_correction_confirmation(llm, query, chunks, embedder, index):
    st.warning(f"Did you mean **{st.session_state.suggested_intent.upper()}**?")

    confirmation = st.text_input("Please answer (yes/no)", key="correction_confirm")

    if not confirmation:
        return

    if confirmation.lower() in ["yes", "y"]:
        confirmed_intent = st.session_state.suggested_intent
        st.session_state.awaiting_correction_confirmation = False

        st.success(f"Great! Proceeding with {confirmed_intent.upper()}")

        if confirmed_intent == "recon":
            st.session_state.recon_stage = "confirm"
            st.warning("Do you want to do reconciliation? (yes / no)")

        elif confirmed_intent == "ocr":
            st.info("OCR feature coming soon!")

        elif confirmed_intent == "kyc":
            st.info("KYC feature coming soon!")

    else:
        st.info("Okay! Treating it as a normal company query.")
        st.session_state.awaiting_correction_confirmation = False

        query = st.session_state.original_query
        results = rag_search(query, chunks, embedder, index)

        response = answer_from_context(llm, query, results)
        st.write(response)


# ============================================================
# 6) MAIN CHAT FLOW
# ============================================================

def handle_convo(llm, query, chunks, embedder, index):
    results = rag_search(query, chunks, embedder, index)
    response = answer_from_context(llm, query, results)
    st.write(response)
    st.session_state.recon_stage = None


def handle_recon_intent():
    st.session_state.recon_stage = "confirm"
    st.warning("Do you want to do reconciliation? (yes / no)")


def handle_ocr_intent():
    st.info("OCR feature coming soon!")
    st.session_state.recon_stage = None


def handle_kyc_intent():
    st.info("KYC feature coming soon!")
    st.session_state.recon_stage = None


def run_main_flow(llm, correction_prompt, intent_prompt, query, chunks, embedder, index):
    if not query:
        return

    if is_greeting(query):
        st.success("Hello 👋 I'm the Technodysis chatbot. How can I help you?")
        st.session_state.recon_stage = None
        return

    # STEP A: Correction detection
    correction_result = detect_possible_correction(llm, correction_prompt, query)

    if correction_result["needs_confirmation"] is True:
        st.session_state.awaiting_correction_confirmation = True
        st.session_state.suggested_intent = correction_result["suggested_intent"]
        st.session_state.original_query = query
        st.rerun()

    # STEP B: Intent classification
    intent = classify_intent(llm, intent_prompt, query)
    st.info(f"Detected intent: **{intent}**")

    if intent == "convo":
        handle_convo(llm, query, chunks, embedder, index)

    elif intent == "recon":
        handle_recon_intent()

    elif intent == "ocr":
        handle_ocr_intent()

    elif intent == "kyc":
        handle_kyc_intent()

    else:
        st.warning("This intent is not handled yet.")


# ============================================================
# 7) RECON FLOW
# ============================================================

def recon_confirm_ui():
    reply = st.text_input("Your response", key="recon_confirm")

    if not reply:
        return

    if reply.lower() in ["yes", "y"]:
        st.session_state.recon_stage = "upload"
    else:
        st.session_state.recon_stage = None
        st.info("Reconciliation cancelled.")


def recon_upload_ui():
    st.success("Please upload the two files for reconciliation")

    file1 = st.file_uploader("Upload first file", type=["pdf", "csv", "xlsx"], key="file1")
    file2 = st.file_uploader("Upload second file", type=["pdf", "csv", "xlsx"], key="file2")

    if file1 and file2:
        st.success("Files uploaded successfully")
        st.write(f"📄 File 1: {file1.name}")
        st.write(f"📄 File 2: {file2.name}")

        # Later:
        # recon_result = recon_agent(file1, file2)
        # st.write(recon_result)


def run_recon_flow():
    if st.session_state.recon_stage == "confirm":
        recon_confirm_ui()

    if st.session_state.recon_stage == "upload":
        recon_upload_ui()


# ============================================================
# 8) APP ENTRY POINT
# ============================================================

def main():
    init_app()
    init_session_state()

    llm = load_llm()

    whisper_model = load_whisper_model()
    chunks, embedder, index = setup_rag()

    correction_prompt = build_correction_prompt(llm)
    intent_prompt = build_intent_prompt()

    query = get_query_from_ui(whisper_model)
    st.session_state.final_query = query

    # If waiting for correction confirmation
    if st.session_state.awaiting_correction_confirmation:
        handle_correction_confirmation(llm, query, chunks, embedder, index)

    else:
        run_main_flow(llm, correction_prompt, intent_prompt, query, chunks, embedder, index)

    # Recon stages always handled at end
    run_recon_flow()


if __name__ == "__main__":
    main()
