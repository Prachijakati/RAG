import streamlit as st
import os
import tempfile
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from docx import Document

import faiss
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from faster_whisper import WhisperModel


# ============================================================
# 0) CONFIG
# ============================================================

TOP1_THRESHOLD = 0.60
GAP_THRESHOLD = 0.20

# OOD threshold:
# - If too high -> it will reject valid queries
# - If too low -> it will allow nonsense
OOD_SIM_THRESHOLD = 0.33

INTENT_DATASET_PATH = "intent_dataset.csv"


# ============================================================
# 0.1) INIT
# ============================================================

def init_app():
    load_dotenv()
    st.set_page_config(page_title="Technodysis Voice Chatbot", layout="centered")
    st.title("🎙️🤖 Technodysis Voice Chatbot")


def init_session_state():
    defaults = {
        "recon_stage": None,
        "final_query": "",

        # intent handling
        "awaiting_intent_choice": False,
        "top3_intents": [],
        "top3_scores": [],
        "query_for_choice": "",

        # selected intent
        "final_intent": None,
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
        device="cpu",  # change to cuda if GPU
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
# 3) INTENT MODEL LOADING
# ============================================================

def clean_text(text: str) -> str:
    text = str(text).strip().lower()
    text = " ".join(text.split())
    return text


@st.cache_resource
def load_intent_model():
    clf = joblib.load("intent_model.pkl")
    le = joblib.load("label_encoder.pkl")

    # Must match training
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    return clf, le, embedder


# ============================================================
# 4) OOD GATE USING TRAINING DATASET + FAISS
# ============================================================

@st.cache_resource
def build_ood_faiss_index(_intent_embedder):
    """
    Builds a FAISS index using the SAME training dataset (intent_dataset.csv).
    This becomes our OOD detection bank.

    We use _intent_embedder to avoid Streamlit hashing issues.
    """
    df = pd.read_csv(INTENT_DATASET_PATH)

    if "text" not in df.columns:
        raise ValueError("intent_dataset.csv must contain a 'text' column")

    texts = df["text"].astype(str).tolist()
    texts = [clean_text(t) for t in texts]

    embs = _intent_embedder.encode(texts, convert_to_numpy=True)
    faiss.normalize_L2(embs)

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    return texts, index


def is_out_of_domain(query: str, embedder, ood_index, threshold=OOD_SIM_THRESHOLD):
    """
    Checks similarity of query to nearest training example.
    If too low -> treat as OOD (nonsense / unrelated).
    """
    query_clean = clean_text(query)

    q_emb = embedder.encode([query_clean], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)

    scores, ids = ood_index.search(q_emb, 1)
    max_sim = float(scores[0][0])

    return max_sim < threshold, max_sim


# ============================================================
# 5) INTENT PREDICTION
# ============================================================

def predict_intent(query: str, clf, le, embedder):
    """
    Returns:
      top_intent: str
      top_confidence: float
      top3_intents: list[str]
      top3_scores: list[float]
      gap: float
    """
    query_clean = clean_text(query)

    emb = embedder.encode([query_clean], convert_to_numpy=True)

    probs = clf.predict_proba(emb)[0]
    class_indices_sorted = np.argsort(probs)[::-1]

    top_idx = class_indices_sorted[0]
    top_conf = float(probs[top_idx])
    top_intent = le.inverse_transform([top_idx])[0]

    top3_idx = class_indices_sorted[:3]
    top3_scores = [float(probs[i]) for i in top3_idx]
    top3_intents = [le.inverse_transform([i])[0] for i in top3_idx]

    top2_conf = float(probs[class_indices_sorted[1]]) if len(class_indices_sorted) > 1 else 0.0
    gap = top_conf - top2_conf

    return top_intent, top_conf, top3_intents, top3_scores, gap


# ============================================================
# 6) UI INPUT
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
# 7) TOP-3 INTENT CHOICE UI
# ============================================================

def intent_label_ui(intent: str) -> str:
    mapping = {
        "recon": "📊 Recon (compare / reconcile 2 files)",
        "ocr": "🖼️ OCR (extract text from document/image)",
        "kyc": "🪪 KYC (identity verification)",
        "convo": "💬 Convo (company questions)"
    }
    return mapping.get(intent, intent)


def show_top3_intent_choice():
    st.warning("⚠️ I'm not fully confident about the intent.")
    st.write("Please choose what you meant:")

    intents = st.session_state.top3_intents
    scores = st.session_state.top3_scores

    options = []
    for intent, score in zip(intents, scores):
        options.append(f"{intent_label_ui(intent)}  —  {int(score * 100)}%")

    choice = st.radio("Select one:", options, key="intent_choice_radio")

    if st.button("Confirm intent"):
        selected_index = options.index(choice)
        selected_intent = intents[selected_index]

        st.session_state.final_intent = selected_intent
        st.session_state.awaiting_intent_choice = False
        st.rerun()


# ============================================================
# 8) INTENT HANDLERS
# ============================================================

def handle_convo(llm, query, chunks, rag_embedder, rag_index):
    results = rag_search(query, chunks, rag_embedder, rag_index)
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


# ============================================================
# 9) RECON FLOW
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


def run_recon_flow():
    if st.session_state.recon_stage == "confirm":
        recon_confirm_ui()

    if st.session_state.recon_stage == "upload":
        recon_upload_ui()


# ============================================================
# 10) MAIN
# ============================================================

def main():
    init_app()
    init_session_state()

    llm = load_llm()
    whisper_model = load_whisper_model()

    chunks, rag_embedder, rag_index = setup_rag()

    clf, le, intent_embedder = load_intent_model()

    # Build OOD FAISS index using training dataset
    ood_texts, ood_index = build_ood_faiss_index(intent_embedder)

    query = get_query_from_ui(whisper_model)
    st.session_state.final_query = query

    if not query:
        run_recon_flow()
        return

    if is_greeting(query):
        st.success("Hello 👋 I'm the Technodysis chatbot. How can I help you?")
        st.session_state.recon_stage = None
        run_recon_flow()
        return

    # If awaiting user selection from top-3
    if st.session_state.awaiting_intent_choice:
        show_top3_intent_choice()
        run_recon_flow()
        return

    # If user already selected
    if st.session_state.final_intent is not None:
        intent = st.session_state.final_intent
        st.session_state.final_intent = None
        confidence = None
        gap = None

    else:
        # 1) OOD gate
        ood, max_sim = is_out_of_domain(
            query=query,
            embedder=intent_embedder,
            ood_index=ood_index,
            threshold=OOD_SIM_THRESHOLD
        )

        if ood:
            st.warning("😅 Sorry, I didn’t understand that. Can you please repeat your query clearly?")
            st.caption(f"(debug: max similarity to known intents = {max_sim:.2f})")
            run_recon_flow()
            return

        # 2) Predict intent
        intent, confidence, top3_intents, top3_scores, gap = predict_intent(
            query=query,
            clf=clf,
            le=le,
            embedder=intent_embedder
        )

        # 3) Confidence routing
        if confidence < TOP1_THRESHOLD and gap < GAP_THRESHOLD:
            st.session_state.awaiting_intent_choice = True
            st.session_state.top3_intents = top3_intents
            st.session_state.top3_scores = top3_scores
            st.session_state.query_for_choice = query
            st.rerun()

    # Show intent info
    if confidence is not None:
        st.info(f"Detected intent: **{intent}** (confidence: {int(confidence*100)}%, gap: {gap:.2f})")
    else:
        st.info(f"Selected intent: **{intent}**")

    # Route
    if intent == "convo":
        handle_convo(llm, query, chunks, rag_embedder, rag_index)

    elif intent == "recon":
        handle_recon_intent()

    elif intent == "ocr":
        handle_ocr_intent()

    elif intent == "kyc":
        handle_kyc_intent()

    else:
        st.warning("This intent is not handled yet.")

    run_recon_flow()


if __name__ == "__main__":
    main()
