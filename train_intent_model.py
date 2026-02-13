import pandas as pd
import numpy as np
import joblib
import os

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


DATA_PATH = "intent_dataset.csv"
MODEL_PATH = "intent_model.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def clean_text(text: str) -> str:
    text = str(text).strip().lower()
    text = " ".join(text.split())
    return text


def main():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found. Run generate_intent_dataset.py first.")
        return

    print("📌 Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    df["text"] = df["text"].apply(clean_text)
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    print("✅ Dataset size:", len(df))
    print(df["label"].value_counts())

    print("\n📌 Loading SentenceTransformer:", EMBEDDING_MODEL_NAME)
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("📌 Creating embeddings...")
    X = embedder.encode(df["text"].tolist(), convert_to_numpy=True)
    y = df["label"].tolist()

    print("📌 Encoding labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("📌 Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print("📌 Training Logistic Regression...")
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    print("\n📌 Evaluating...")
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("✅ Accuracy:", acc)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n📌 Saving model + label encoder...")
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)

    print("✅ Saved:", MODEL_PATH)
    print("✅ Saved:", LABEL_ENCODER_PATH)

    print("\n🎉 Training complete!")


if __name__ == "__main__":
    main()