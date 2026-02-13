import streamlit as st
import os
import tempfile
import joblib
import pandas as pd
import difflib
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

# Try loading Whisper (Optional)
try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

# ============================================================
# 0) CONFIGURATION
# ============================================================

# Files
MODEL_PATH = "intent_model.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
INTENT_DATASET_PATH = "intent_dataset.csv"

# --- THRESHOLDS ---
FUZZY_MIN_THRESHOLD = 0.60  # For typos (obr -> ocr)
ML_AUTO_THRESHOLD = 0.65    # If ML is > 65% confident, just do it.
OOD_THRESHOLD = 0.40        # Below this = Gibberish

# Stopwords to ignore
IGNORE_TOKENS = {
    "for", "and", "the", "but", "not", "are", "you", "can", "with", "do", "to", "my", "in", "of", "is", "it",
    "i", "me", "we", "us", "this", "that", "on", "at", "from", "by", "an", "be",
    "want", "need", "would", "could", "should", "will", "perform", "please", "help",
    "make", "give", "show", "tell", "say", "find", "search", "looking", "process"
}

# KEYWORDS (Clean List - NO TYPOS)
INTENT_KEYWORDS = {
    "ocr": ["ocr", "optical", "extract", "scanning", "scan", "image", "text", "read", "convert"],
    "kyc": ["kyc", "know", "customer", "verification", "identity", "id", "card", "passport", "aadhaar", "pan", "driving", "license"],
    "recon": ["recon", "reconciliation", "reconcile", "comparing", "match", "compare", "files", "excel", "csv", "mismatch", "differences"],
    "convo": ["convo", "company", "question", "ask", "chat", "tell", "about", "policy", "what", "how", "who", "founder", "services", "contact"]
}

# ============================================================
# 1) INITIALIZATION
# ============================================================

def init_app():
    st.set_page_config(page_title="Technodysis Voice Chatbot", layout="centered")
    st.title("🎙️🤖 Technodysis Voice Chatbot")

def init_session_state():
    defaults = {
        "last_query": "",
        "suggested_intent": None, # For "Did you mean?"
        "final_intent": None,     # For execution
        "top3_choices": None,     # For vague queries
        "debug_logs": [],
        "probs_df": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_PATH):
        return None, None, None, None
    
    clf = joblib.load(MODEL_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    ood_index = None
    if os.path.exists(INTENT_DATASET_PATH):
        df = pd.read_csv(INTENT_DATASET_PATH)
        corpus = df["text"].astype(str).tolist()
        corpus_embeddings = embedder.encode(corpus, convert_to_numpy=True)
        faiss.normalize_L2(corpus_embeddings)
        ood_index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
        ood_index.add(corpus_embeddings)
    
    return clf, le, embedder, ood_index

@st.cache_resource
def load_whisper():
    if WhisperModel:
        return WhisperModel("base", device="cpu", compute_type="int8")
    return None

# ============================================================
# 2) LOGIC CORE
# ============================================================

def clean_text(text: str) -> str:
    text = str(text).strip().lower()
    return " ".join(text.split())

def is_out_of_domain(query, embedder, index):
    if index is None: return False, 1.0
    q_emb = embedder.encode([clean_text(query)], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k=1)
    similarity = float(D[0][0])
    return similarity < OOD_THRESHOLD, similarity

def analyze_intent(query, clf, le, embedder, ood_index):
    logs = []
    clean_query = clean_text(query)
    tokens = clean_query.split()
    meaningful_tokens = [t for t in tokens if t not in IGNORE_TOKENS]
    
    logs.append(f"**1. Input Analysis**")
    logs.append(f"- Raw: `{query}`")

    # --- A. OOD Check ---
    is_ood, ood_score = is_out_of_domain(query, embedder, ood_index)
    if is_ood:
        logs.append(f"❌ OOD Detected (Score: {ood_score:.2f})")
        return "ood", None, logs, None

    # --- B. Keyword Analysis (Exact vs Fuzzy) ---
    logs.append(f"\n**2. Keyword Analysis**")
    
    # 1. Exact Match -> Auto Execute
    for intent, keywords in INTENT_KEYWORDS.items():
        if set(meaningful_tokens) & set(keywords):
            logs.append(f"✅ Exact Match: {intent.upper()}")
            return "auto_execute", intent, logs, None

    # 2. Fuzzy Match -> "Did you mean?"
    best_fuzzy_intent = None
    best_fuzzy_score = 0.0
    for intent, keywords in INTENT_KEYWORDS.items():
        for token in meaningful_tokens:
            matches = difflib.get_close_matches(token, keywords, n=1, cutoff=FUZZY_MIN_THRESHOLD)
            if matches:
                ratio = difflib.SequenceMatcher(None, token, matches[0]).ratio()
                if ratio > best_fuzzy_score:
                    best_fuzzy_score = ratio
                    best_fuzzy_intent = intent

    if best_fuzzy_intent:
        logs.append(f"⚠️ Fuzzy Match: {best_fuzzy_intent} (Score: {best_fuzzy_score:.2f})")
        return "fuzzy_suggest", best_fuzzy_intent, logs, None

    # --- C. ML Model Prediction ---
    logs.append(f"\n**3. ML Model Prediction**")
    emb = embedder.encode([clean_query], convert_to_numpy=True)
    probs = clf.predict_proba(emb)[0]
    
    # Create sorted list of (Intent, Score)
    top_indices = np.argsort(probs)[::-1]
    top_intent = le.classes_[top_indices[0]]
    top_score = probs[top_indices[0]]
    
    # Store top 3 for the UI
    top3_data = []
    for i in range(3):
        idx = top_indices[i]
        top3_data.append((le.classes_[idx], probs[idx]))
    
    logs.append(f"- Top ML: {top_intent} ({top_score:.2f})")
    
    # Decision:
    if top_score > ML_AUTO_THRESHOLD:
        logs.append("✅ ML Confidence High -> Auto Execute")
        return "auto_execute", top_intent, logs, top3_data
    else:
        logs.append("❓ ML Confidence Low -> Show Top 3")
        return "ambiguous_top3", top3_data, logs, top3_data


# ============================================================
# 3) UI MAIN
# ============================================================

def main():
    init_app()
    init_session_state()
    
    clf, le, embedder, ood_index = load_models()
    whisper = load_whisper()
    
    if not clf:
        st.error("⚠️ Models not found!")
        st.stop()

    # --- Input Section ---
    st.subheader("Choose Input Method")
    input_type = st.radio("Mode:", ["🔴 ⌨ Type", "🔴 🎙 Voice"], horizontal=True)
    
    user_query = ""
    if "Type" in input_type:
        user_query = st.text_input("Ask something", placeholder="e.g., I want to do obr").strip()
    else:
        audio = st.audio_input("Record")
        if audio and whisper:
            with st.spinner("Transcribing..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    f.write(audio.getvalue())
                    fname = f.name
                segments, _ = whisper.transcribe(fname)
                user_query = " ".join([s.text for s in segments])
                st.write(f"**You said:** {user_query}")
                os.remove(fname)

    # --- Processing ---
    if user_query and user_query != st.session_state.last_query:
        st.session_state.last_query = user_query
        
        # Reset all decision states
        st.session_state.final_intent = None
        st.session_state.suggested_intent = None
        st.session_state.top3_choices = None
        
        decision, data, logs, _ = analyze_intent(user_query, clf, le, embedder, ood_index)
        st.session_state.debug_logs = logs
        
        if decision == "ood":
            st.error("😕 Sorry, I didn't understand that. Please try again.")
        elif decision == "auto_execute":
            st.session_state.final_intent = data # data is intent_string
            st.rerun()
        elif decision == "fuzzy_suggest":
            st.session_state.suggested_intent = data # data is intent_string
        elif decision == "ambiguous_top3":
            st.session_state.top3_choices = data # data is list of (intent, score)

    # --- Debug Panel ---
    if st.session_state.debug_logs:
        with st.expander("🧠 Logic Logs", expanded=False):
            st.write(st.session_state.debug_logs)

    # --- 1. "Did you mean?" UI (Fuzzy) ---
    if st.session_state.suggested_intent:
        st.info(f"🧐 I noticed a word similar to **{st.session_state.suggested_intent.upper()}**.")
        st.write(f"Did you mean to perform {st.session_state.suggested_intent.upper()}?")
        
        c1, c2 = st.columns(2)
        if c1.button("✅ Yes"):
            st.session_state.final_intent = st.session_state.suggested_intent
            st.session_state.suggested_intent = None
            st.rerun()
        if c2.button("❌ No"):
            st.session_state.suggested_intent = None
            st.rerun()

    # --- 2. "Top 3 Choices" UI (Ambiguous ML) ---
    if st.session_state.top3_choices:
        st.warning("⚠️ I'm not fully confident. Please choose your intent:")
        
        # Format options for Radio button
        options = [f"{intent.upper()} ({int(score*100)}%)" for intent, score in st.session_state.top3_choices]
        choice = st.radio("Select one:", options)
        
        if st.button("Confirm Selection"):
            # Extract intent string from selection (e.g., "OCR (45%)" -> "ocr")
            selected_intent = choice.split(" ")[0].lower()
            st.session_state.final_intent = selected_intent
            st.session_state.top3_choices = None
            st.rerun()

    # --- Final Execution ---
    if st.session_state.final_intent:
        intent = st.session_state.final_intent
        
        st.markdown(f"""
        <div style="background-color:#d4edda;padding:10px;border-radius:5px;border:1px solid #c3e6cb;">
            <h3 style="color:#155724;margin:0;">🚀 Executing: {intent.upper()}</h3>
        </div><br>
        """, unsafe_allow_html=True)
        
        if intent == "ocr":
            st.info("📄 **OCR Module**: Upload a document.")
            st.file_uploader("Upload", key="ocr")
        elif intent == "recon":
            st.info("📊 **Recon Module**: Upload two files.")
            c1, c2 = st.columns(2)
            c1.file_uploader("File A", key="r1")
            c2.file_uploader("File B", key="r2")
        elif intent == "kyc":
            st.info("🪪 **KYC Module**: Identity verification.")
            st.button("Start Verification")
        elif intent == "convo":
            st.info("🤖 **Chat**: How can I help you?")

if __name__ == "__main__":
    main()