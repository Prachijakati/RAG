import pandas as pd
import random

random.seed(42)

def build_recon():
    # --- Standard Sentences ---
    actions = [
        "reconcile", "compare", "match", "cross check", "cross-check",
        "verify", "validate", "check", "find mismatch in", "highlight differences in"
    ]
    objects = [
        "two files", "two spreadsheets", "two excel files", "two csv files",
        "two reports", "two statements", "bank statement and ledger",
        "invoice report and payment report", "transaction records", "payment records"
    ]
    extras = [
        "", "for me", "today", "quickly", "in detail", "and show mismatches",
        "and show missing entries", "and show duplicates", "and tell me what changed"
    ]
    
    # --- SHORT / LAZY / TYPO Commands (New) ---
    lazy_prompts = [
        "recon", "do recon", "reconcile", "compare files", "match files",
        "check mismatch", "find diffs", "excel compare", "csv match",
        "recom", "do recom", "perform recom" # Typos included
    ]

    samples = []
    # 250 Complex sentences
    for _ in range(250):
        s = f"{random.choice(actions)} {random.choice(objects)} {random.choice(extras)}"
        samples.append(s.strip())
    
    # 100 Short/Lazy sentences (Crucial for robust detection)
    for _ in range(100):
        samples.append(random.choice(lazy_prompts))
        
    return samples


def build_ocr():
    # --- Standard Sentences ---
    actions = [
        "extract text from", "perform ocr on", "read text from", "scan",
        "convert", "detect text in", "recognize text from"
    ]
    objects = [
        "this image", "this photo", "this document", "this scanned document",
        "this pdf", "this scanned pdf", "this invoice", "this receipt",
        "this bill", "this form"
    ]
    extras = [
        "", "and give me the content", "and return plain text",
        "and copy the text", "and show me the extracted words"
    ]

    # --- SHORT / LAZY / TYPO Commands (New) ---
    lazy_prompts = [
        "ocr", "do ocr", "ocr this", "ocr scan", "extract text", 
        "text extract", "read this", "convert to text", "scan to text",
        "ocr my file", "get text", "grab text", "ocr image",
        "obr", "do obr", "odr", "do odr", "perform odr" # Typos explicitly added
    ]

    samples = []
    for _ in range(250):
        s = f"{random.choice(actions)} {random.choice(objects)} {random.choice(extras)}"
        samples.append(s.strip())

    for _ in range(100):
        samples.append(random.choice(lazy_prompts))
        
    return samples


def build_kyc():
    # --- Standard Sentences ---
    actions = [
        "verify", "validate", "do kyc for", "perform kyc for",
        "check identity for", "confirm identity for"
    ]
    objects = [
        "aadhaar", "aadhar", "pan card", "passport", "driving license",
        "voter id", "government id", "customer", "user"
    ]
    extras = [
        "", "for onboarding", "for account opening", "for verification",
        "and extract name and dob", "and validate details"
    ]

    # --- SHORT / LAZY Commands (New) ---
    lazy_prompts = [
        "kyc", "do kyc", "check id", "verify id", "verify customer",
        "aadhaar check", "pan check", "passport check", "validate user",
        "know your custard" # funny typo/autocorrect case
    ]

    samples = []
    for _ in range(250):
        s = f"{random.choice(actions)} {random.choice(objects)} {random.choice(extras)}"
        samples.append(s.strip())

    for _ in range(100):
        samples.append(random.choice(lazy_prompts))

    return samples


def build_convo():
    templates = [
        "tell me about technodysis",
        "what is technodysis",
        "what services does technodysis provide",
        "who is the founder of technodysis",
        "where is technodysis located",
        "what solutions do you offer",
        "do you provide ai services",
        "do you provide automation services",
        "how can i contact technodysis",
        "explain your company",
        "what does your company do",
        "what projects have you done",
        "what industries do you work with"
    ]

    greetings = ["hi", "hello", "hey", "good morning", "good evening"]

    samples = []
    for _ in range(200):
        if random.random() < 0.25:
            samples.append(random.choice(greetings))
        else:
            samples.append(random.choice(templates))
    return samples


def main():
    rows = []

    for t in build_recon():
        rows.append({"text": t, "label": "recon"})

    for t in build_ocr():
        rows.append({"text": t, "label": "ocr"})

    for t in build_kyc():
        rows.append({"text": t, "label": "kyc"})

    for t in build_convo():
        rows.append({"text": t, "label": "convo"})

    df = pd.DataFrame(rows)
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df.to_csv("intent_dataset.csv", index=False)
    print("✅ intent_dataset.csv generated!")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()