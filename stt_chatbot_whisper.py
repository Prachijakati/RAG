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


# ============================================================
# 0) CONFIG + INIT
# ============================================================

def init_app():
    load_dotenv()
    st.set_page_config(page_title="Technodysis Voice Chatbot", layout="centered")
    st.title("🎙️🤖 Technodysis Voice Chatbot")


def init_session_state():
    defaults = {
        "recon_stage": None,

        # intent decision
        "final_query": "",
        "final_intent": None,

        # ambiguity handler
        "awaiting_disambiguation": False,
        "intent_candidates": [],  # list of tuples (intent, score)

        # for UI reruns
        "last_query": ""
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
# 3) EMBEDDING-BASED INTENT CLASSIFIER (NO PROMPTS)
# ============================================================

def get_intent_examples():
    """
    IMPORTANT:
    Add more examples here over time.
    This is how accuracy becomes excellent.
    """

    def get_intent_examples():
        return {
            "recon": [
                # --- Direct recon requests ---
                "I want to do recon",
                "I want to perform recon",
                "Can you do recon for me?",
                "Do reconciliation",
                "Start reconciliation",
                "I need reconciliation",
                "Help me with reconciliation",
                "I want to reconcile files",
                "Reconcile two files",
                "Reconcile these documents",
                "Reconcile these records",
                "Reconcile the data",
                "Reconcile my transactions",
                "Reconcile my statements",
                "Reconcile bank transactions",
                "Reconcile invoice data",
                "Reconcile payment data",
                "Reconcile accounts",

                # --- Compare 2 files ---
                "Compare these two files",
                "Compare these files",
                "Compare two spreadsheets",
                "Compare these two spreadsheets",
                "Compare two Excel sheets",
                "Compare these two Excel sheets",
                "Compare two CSV files",
                "Compare these two CSV files",
                "Compare two PDF reports",
                "Compare these two PDF reports",
                "Compare these two reports",
                "Compare file 1 and file 2",
                "Check differences between two files",
                "Check mismatch between two files",
                "Find differences between these files",
                "Find mismatch between these reports",
                "Detect differences in these spreadsheets",
                "Show differences between the two documents",
                "Tell me what changed between these two files",

                # --- Matching / cross-checking ---
                "Match transactions between two statements",
                "Match the data between two files",
                "Cross check these two files",
                "Cross-check these two files",
                "Cross verify these two reports",
                "Cross-verify these two reports",
                "Validate these two datasets against each other",
                "Compare and validate these two files",
                "Check whether both files have same data",
                "Check if these files match",
                "Check if these statements match",
                "Ensure both reports match",
                "Make sure these two files match",

                # --- Bank/finance specific ---
                "Verify my bank transaction records",
                "I want to verify my bank transaction records",
                "Compare bank statement with ledger",
                "Compare ledger and bank statement",
                "Reconcile bank statement with company ledger",
                "Check bank statement vs accounting report",
                "Compare bank statement and payment report",
                "Check if bank statement matches invoices",
                "Match UTR numbers between two files",
                "Compare UTR entries between two reports",
                "Check missing transactions in bank statement",
                "Find missing entries in my transaction report",
                "Detect duplicate transactions",
                "Find duplicate entries in statement",
                "Find unmatched transactions",

                # --- Invoice / billing specific ---
                "Compare invoices between two files",
                "Reconcile invoice report with payment report",
                "Match invoices with payments",
                "Check invoice vs payment mismatch",
                "Find missing invoices",
                "Find unmatched invoices",
                "Compare billing reports",
                "Compare purchase records and invoice report",

                # --- Payroll / HR style ---
                "Compare payroll sheet with bank transfer sheet",
                "Match salary payments with payroll report",
                "Reconcile payroll data",

                # --- Casual / human phrasing ---
                "Can you check these two files and tell me the difference?",
                "Can you tell me what is missing between these two files?",
                "Please check which entries are not matching",
                "Please compare and highlight mismatches",
                "I uploaded two files, compare them",
                "I have two reports, check mismatches",
                "I have two sheets, match them",
                "I have two statements, reconcile them",
                "I need to compare two documents and see mismatches",
                "Check my transactions and tell me what is missing",
                "Check if anything is different in these files",
                "Do a reconciliation for these two files",

                # --- Short / noisy speech-to-text type ---
                "recon",
                "reconcile",
                "reconciliation",
                "do recon",
                "do reconciliation",
                "compare files",
                "match files",
                "check mismatch",
                "bank recon",
                "statement recon",
            ],

            "ocr": [
                # --- Direct OCR requests ---
                "I want to do OCR",
                "Perform OCR",
                "Start OCR",
                "Can you do OCR for me?",
                "Extract text from this image",
                "Extract text from this photo",
                "Extract text from this document",
                "Extract text from this PDF",
                "Extract text from this scanned document",
                "Read text from this image",
                "Read text from this photo",
                "Read text from this document",
                "Read text from this scanned page",
                "Convert image to text",
                "Convert photo to text",
                "Convert scanned document into text",
                "Convert PDF to text",

                # --- Human phrasing ---
                "Can you scan this and give me the text?",
                "Can you read this document and extract the content?",
                "I have an image, I want the text from it",
                "I have a photo, extract the words from it",
                "I want the text written inside this image",
                "Please detect the text in this picture",
                "Please recognize the text in this image",
                "Please pull out the text from this document",
                "Please read the text from the file",
                "I want to copy the text from this image",
                "Help me extract words from this document",

                # --- PDF / scanned forms ---
                "Extract text from a scanned PDF",
                "Read a scanned PDF and give me the text",
                "Convert scanned PDF to text",
                "Extract text from a form",
                "Read this form and extract text",
                "Extract text from a bill",
                "Extract text from a receipt",
                "Extract text from an invoice",
                "Read receipt and extract details",
                "Read invoice and extract details",

                # --- OCR definition ---
                "Optical Character Recognition",
                "OCR extraction",
                "Text recognition",
                "Image text extraction",
                "Document text extraction",

                # --- Speech-to-text style mistakes ---
                "I want to do ocr",
                "do ocr",
                "perform ocr",
                "extract text",
                "scan and extract",
                "read the image",
            ],

            "kyc": [
                # --- Direct KYC requests ---
                "I want to do KYC",
                "Perform KYC",
                "Start KYC verification",
                "I want KYC verification",
                "Do KYC verification using aadhar",
                "KYC verification for onboarding",
                "Verify customer identity",
                "Validate customer identity",
                "Identity verification",
                "Verify identity",
                "Verify a person using ID card",

                # --- Documents ---
                "Verify Aadhaar card",
                "Verify PAN card",
                "Verify passport",
                "Verify driving license",
                "Verify voter ID",
                "Verify government ID",
                "Verify ID proof",
                "Check identity document",
                "Validate ID document",
                "Extract details from Aadhaar",
                "Extract details from PAN card",
                "Extract details from passport",
                "Extract name and DOB from Aadhaar",
                "Extract name and date of birth from passport",
                "Read Aadhaar and verify details",
                "Read PAN card and verify details",

                # --- Human phrasing ---
                "I want to verify my aadhar card",
                "Can you verify my passport?",
                "Can you check if this ID is valid?",
                "I want to confirm the person details from ID",
                "I want to check identity of a customer",
                "I want to do customer onboarding verification",
                "I want to verify KYC documents",
                "Help me verify identity documents",
                "Please verify the ID details",

                # --- Banking / onboarding style ---
                "KYC for bank account opening",
                "KYC for customer onboarding",
                "KYC for new user registration",
                "Verify user details using Aadhaar",
                "Check if Aadhaar details match",
                "Validate name and DOB from ID",

                # --- Short / speech style ---
                "kyc",
                "k y c",
                "do kyc",
                "verify identity",
                "identity check",
                "aadhar verification",
                "passport verification",
            ],

            "convo": [
                # --- Company related ---
                "Tell me about Technodysis",
                "What is Technodysis?",
                "What services does Technodysis provide?",
                "What does Technodysis do?",
                "What are the services offered by Technodysis?",
                "Explain Technodysis services",
                "What kind of AI services do you provide?",
                "Do you provide AI and ML services?",
                "Do you provide automation services?",
                "What solutions do you offer?",
                "What industries do you work with?",
                "Who is the founder of Technodysis?",
                "Who started Technodysis?",
                "Who is the CEO of Technodysis?",
                "Where is your office located?",
                "What is your office address?",
                "How can I contact Technodysis?",
                "Do you have a website?",
                "Do you have a LinkedIn page?",
                "How many employees do you have?",
                "What projects have you done?",
                "Tell me about your company background",
                "Give me a summary of Technodysis",

                # --- Generic chat ---
                "Hi",
                "Hello",
                "Hey",
                "How are you?",
                "Tell me a joke",
                "What can you do?",
                "How does this chatbot work?",
                "Help me understand your features",
                "What are your capabilities?",
                "What is this application for?",

                # --- Random unrelated questions ---
                "What is the weather today?",
                "Who is the prime minister of India?",
                "Explain machine learning",
                "What is OCR?",
                "What is KYC?",
                "What is reconciliation?",
            ]
        }


@st.cache_resource
def build_intent_embedding_index():
    """
    Builds an embedding database for intent examples.
    """
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    intent_examples = get_intent_examples()

    all_texts = []
    all_labels = []

    for intent, examples in intent_examples.items():
        for ex in examples:
            all_texts.append(ex)
            all_labels.append(intent)

    embs = embedder.encode(all_texts, convert_to_numpy=True)
    faiss.normalize_L2(embs)

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    return embedder, index, all_labels, all_texts


def predict_intent_with_ambiguity(query, embedder, index, labels, texts,
                                  top_k=6,
                                  confidence_threshold=0.35,
                                  ambiguity_gap=0.06):
    """
    Returns:
      - final_intent (or None if ambiguous)
      - candidates list [(intent, score), ...] sorted by score
    """

    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)

    scores, ids = index.search(q_emb, top_k)

    # Collect top hits
    hits = []
    for idx, score in zip(ids[0], scores[0]):
        hits.append((labels[idx], float(score), texts[idx]))

    # Aggregate best score per intent
    best_per_intent = {}
    for intent, score, ex_text in hits:
        if intent not in best_per_intent or score > best_per_intent[intent]:
            best_per_intent[intent] = score

    # Sort intents by score
    candidates = sorted(best_per_intent.items(), key=lambda x: x[1], reverse=True)

    # If nothing confident -> unknown -> treat as convo
    if not candidates or candidates[0][1] < confidence_threshold:
        return "convo", candidates

    # Ambiguity check
    if len(candidates) >= 2:
        top_intent, top_score = candidates[0]
        second_intent, second_score = candidates[1]

        # If the scores are too close -> ambiguous
        if (top_score - second_score) < ambiguity_gap:
            return None, candidates

    return candidates[0][0], candidates


# ============================================================
# 4) UI INPUT
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
# 5) DISAMBIGUATION UI (THE KEY IMPROVEMENT)
# ============================================================

def show_disambiguation_ui(query):
    st.warning("⚠️ Your question could mean more than one thing.")
    st.write("Please choose what you actually meant:")

    options = [intent for intent, score in st.session_state.intent_candidates[:3]]

    # make it nicer for user
    label_map = {
        "recon": "📊 Reconciliation (compare 2 files / transactions)",
        "ocr": "🖼️ OCR (extract text from image/document)",
        "kyc": "🪪 KYC (identity verification)",
        "convo": "💬 Normal question (company info)"
    }

    display_options = [label_map.get(o, o) for o in options]

    choice = st.radio("Select one:", display_options, key="intent_choice")

    if st.button("Confirm intent"):
        # convert back to internal intent name
        chosen_intent = None
        for k, v in label_map.items():
            if v == choice:
                chosen_intent = k
                break

        st.session_state.final_intent = chosen_intent
        st.session_state.awaiting_disambiguation = False
        st.rerun()


# ============================================================
# 6) MAIN HANDLERS
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
    chunks, rag_embedder, rag_index = setup_rag()

    # Intent classifier embedding index
    intent_embedder, intent_index, intent_labels, intent_texts = build_intent_embedding_index()

    # UI query
    query = get_query_from_ui(whisper_model)
    st.session_state.final_query = query

    if not query:
        run_recon_flow()
        return

    # Greeting
    if is_greeting(query):
        st.success("Hello 👋 I'm the Technodysis chatbot. How can I help you?")
        st.session_state.recon_stage = None
        run_recon_flow()
        return

    # If already disambiguating
    if st.session_state.awaiting_disambiguation:
        show_disambiguation_ui(query)
        run_recon_flow()
        return

    # If user already selected final intent
    if st.session_state.final_intent is not None:
        intent = st.session_state.final_intent
    else:
        # Predict intent using embeddings
        intent, candidates = predict_intent_with_ambiguity(
            query=query,
            embedder=intent_embedder,
            index=intent_index,
            labels=intent_labels,
            texts=intent_texts
        )

        # If ambiguous, ask user
        if intent is None:
            st.session_state.awaiting_disambiguation = True
            st.session_state.intent_candidates = candidates
            st.rerun()

    st.info(f"Detected intent: **{intent}**")

    # Reset after final decision
    st.session_state.final_intent = None

    # Handle intent
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
