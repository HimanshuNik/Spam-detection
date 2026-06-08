"""
Spam Detection — Model Training Script
========================================
Downloads the SMS Spam Collection dataset, preprocesses text with NLP,
trains three classifiers (Naive Bayes, Logistic Regression, Random Forest),
compares their performance, and saves the best-performing model.

Usage:
    python train_model.py
"""

import os
import re
import pickle
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- NLTK Setup ---
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATASET_PATH = os.path.join(BASE_DIR, "spam.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")


# ---------------------------------------------------------------------------
# Dataset Acquisition
# ---------------------------------------------------------------------------
def download_dataset():
    """Download the SMS Spam Collection from UCI, or fall back to a built-in sample."""
    if os.path.exists(DATASET_PATH):
        print(f"[OK] Dataset already exists at {DATASET_PATH}")
        return

    print("[*] Downloading SMS Spam Collection dataset ...")
    zip_path = os.path.join(BASE_DIR, "smsspamcollection.zip")

    try:
        urllib.request.urlretrieve(DATASET_URL, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(BASE_DIR)

        # The extracted file is tab-separated with no header
        tsv_path = os.path.join(BASE_DIR, "SMSSpamCollection")
        df = pd.read_csv(tsv_path, sep="\t", header=None,
                         names=["label", "message"], encoding="latin-1")
        df.to_csv(DATASET_PATH, index=False)

        # Clean up temporary files
        for tmp in [zip_path, tsv_path, os.path.join(BASE_DIR, "readme")]:
            if os.path.exists(tmp):
                os.remove(tmp)

        print(f"[OK] Dataset saved - {len(df)} messages")

    except Exception as exc:
        print(f"[!] Download failed: {exc}")
        print("[*] Creating built-in sample dataset ...")
        _create_fallback_dataset()


def _create_fallback_dataset():
    """Create a small but representative dataset when download is unavailable."""
    spam_msgs = [
        "WINNER!! As a valued network customer you have been selected to receive a 900 prize reward!",
        "Free entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121",
        "URGENT! You have won a 1 week free membership in our 100000 Prize Jackpot!",
        "Congratulations! You've won a free cruise to the Bahamas! Call now to claim!",
        "You are a winner! Text WIN to 12345 to claim your free iPhone today!",
        "CASH PRIZE! Claim your 5000 award now. Call 08001234567. Valid 24hrs only",
        "FREE ringtone! Reply YES to 45678 and get unlimited free ringtones!",
        "Your account has been compromised! Click here to verify your identity now!",
        "Hot singles in your area are waiting! Sign up FREE at hotsingles.com",
        "Earn $5000 per week from home! No experience needed! Visit earn-money-now.com",
        "WINNER! You have been selected for our exclusive loyalty reward of 1000 cash!",
        "Act now! Limited time offer - get 90% discount on all products. Click here!",
        "Your mobile number has won a prize of $10000! To claim call 09061234567",
        "FREE msg: We tried to call you. You have won a prize. Please call back 08001111",
        "Congratulations ur awarded 500 of CD vouchers. Call 09066612661 to collect",
        "You have been chosen to receive a special bonus credit of 500! Call now!",
        "URGENT: Your bank account has been locked. Verify at http://fake-bank.com",
        "Win cash and prizes! Text PLAY to 67890. Costs 1.50 per msg. 18+ only",
        "Double your income with our proven system! No risk! Visit money-doubler.com",
        "FREE entry to our weekly prize draw! Text FREE to 87654 now!",
        "Your parcel is waiting for delivery. Pay 1.99 fee at: http://fake-delivery.com",
        "Claim your guaranteed cash prize! Call 09061743810 before midnight tonight!",
        "You've been selected for a free trial of our premium service! Text YES to 12345",
        "IMPORTANT: We detected unusual activity on your account. Click to secure it now",
        "WIN a brand new car! Enter our competition by texting CAR to 80101. T&Cs apply",
    ]
    ham_msgs = [
        "Hey, are you coming to the party tonight? Let me know so I can pick you up.",
        "I'll be late for dinner. Got stuck in traffic. Save some food for me please!",
        "Can you send me the notes from yesterday's meeting? I missed the last part.",
        "Happy birthday! Hope you have an amazing day. Let's catch up this weekend!",
        "Just finished the project report. I'll email it to you by end of day.",
        "Running late. Will be there in 20 minutes. Start without me if you need to.",
        "Thanks for the help yesterday. I really appreciate it. Coffee on me next time!",
        "Don't forget we have a team meeting at 3pm today. See you there.",
        "Hey! How was your weekend? Did you go hiking like you planned?",
        "Could you pick up some milk on your way home? We're completely out.",
        "I saw the movie last night. It was amazing! We should watch it together sometime.",
        "The weather is beautiful today. Want to go for a walk in the park after work?",
        "I just got the test results back. Everything looks good! Thanks for checking in.",
        "Remember to bring the documents tomorrow. I left them on the kitchen counter.",
        "Are you free this Saturday? My sister is having a barbecue and you're invited.",
        "The train is delayed by 30 minutes. I'll text you when I arrive at the station.",
        "Good morning! Just wanted to say I hope you have a great day today.",
        "Did you finish reading that book I lent you? I'd love to hear what you thought.",
        "I'm at the store. Do we need anything else besides bread and eggs?",
        "Sorry I missed your call. Was in a meeting. Can I call you back in an hour?",
        "Just booked the restaurant for Friday night. Table for 4 at 7pm. Sound good?",
        "The kids had a great time at school today. They made paintings for us!",
        "Heading to the gym now. Want to join? We could grab smoothies after.",
        "Thank you for the birthday wishes! It means a lot to me. Hugs!",
        "I passed the exam! All that studying paid off. Let's celebrate this weekend!",
        "Can you water the plants while I'm away? The ones in the living room need it daily.",
        "Lunch was great today. We should go to that restaurant more often!",
        "My flight lands at 6pm. Can you pick me up from the airport?",
        "Great news — the landlord agreed to fix the leaking tap this week!",
        "I left my umbrella at your place. Can I grab it when I come over tomorrow?",
        "How's the new job going? Hope the first week wasn't too stressful!",
        "Mum says hi and wants to know when you're visiting next.",
        "I finished cooking dinner. Come down whenever you're ready!",
        "Are you still up for the concert next month? Tickets go on sale Friday.",
        "Just wanted to check in and see how you're doing. Miss our chats!",
        "I'll send you the photos from the trip when I get home tonight.",
        "Please remind me to call the dentist tomorrow morning. I keep forgetting!",
        "Loved the recipe you shared. I tried it tonight and it turned out great!",
        "Need to reschedule our coffee date. How about Thursday instead?",
        "We made it home safe. Thanks again for a wonderful evening!",
        "Did you see the sunset tonight? Absolutely gorgeous!",
        "I submitted the assignment early. Fingers crossed for a good grade!",
        "The dog learned a new trick today! I'll show you when you come over.",
        "Let's plan a road trip for the long weekend. Any destination ideas?",
        "Good night! Talk to you tomorrow. Sweet dreams!",
        "I'm making spaghetti for dinner tonight. Should I make extra for you?",
        "Have you heard from Jake recently? We should all hang out soon.",
        "Class got cancelled today, so I'm free all afternoon if you want to meet up.",
        "Just saw your message. Yes, that time works perfectly for me!",
        "The new season of that show we watch just dropped! Want to binge it this weekend?",
    ]

    data = (
        [{"label": "spam", "message": m} for m in spam_msgs]
        + [{"label": "ham", "message": m} for m in ham_msgs]
    )
    df = pd.DataFrame(data)
    df.to_csv(DATASET_PATH, index=False)
    print(f"[OK] Fallback dataset created - {len(df)} messages")


# ---------------------------------------------------------------------------
# NLP Preprocessing
# ---------------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline:
      1. Lowercase
      2. Remove URLs, numbers, punctuation
      3. Tokenize
      4. Remove stopwords
      5. Lemmatize
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)       # URLs
    text = re.sub(r"\d+", "", text)                   # numbers
    text = re.sub(r"[^\w\s]", "", text)               # punctuation
    text = re.sub(r"\s+", " ", text).strip()          # whitespace

    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words and len(t) > 1
    ]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Training & Comparison
# ---------------------------------------------------------------------------
def train_and_compare():
    """Train NB, LR, RF; compare accuracy; save the winner."""

    # ---- Load ----
    print("\n[*] Loading dataset ...")
    df = pd.read_csv(DATASET_PATH, encoding="latin-1")

    # Normalise columns (handles both UCI raw and our CSV)
    if "label" not in df.columns:
        df.columns = ["label", "message"] + list(df.columns[2:])
    df = df[["label", "message"]].dropna()
    print(f"[OK] {len(df)} messages  |  {df['label'].value_counts().to_dict()}")

    # ---- Preprocess ----
    print("[*] Preprocessing text ...")
    df["processed"] = df["message"].apply(preprocess_text)

    # Encode: spam → 1, ham → 0
    df["label_enc"] = (df["label"].str.lower() == "spam").astype(int)

    # ---- Split ----
    X_train, X_test, y_train, y_test = train_test_split(
        df["processed"], df["label_enc"],
        test_size=0.2, random_state=42, stratify=df["label_enc"],
    )

    # ---- TF-IDF ----
    print("[*] Building TF-IDF features (max 5 000, bigrams) ...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # ---- Models ----
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }

    best_model = None
    best_acc = 0.0
    best_name = ""

    print("\n" + "=" * 62)
    print("  MODEL COMPARISON RESULTS")
    print("=" * 62)

    for name, clf in models.items():
        clf.fit(X_train_vec, y_train)
        preds = clf.predict(X_test_vec)

        acc  = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec  = recall_score(y_test, preds, zero_division=0)
        f1   = f1_score(y_test, preds, zero_division=0)

        print(f"\n  {name}")
        print(f"    Accuracy : {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall   : {rec:.4f}")
        print(f"    F1-Score : {f1:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = clf
            best_name = name

    print(f"\n{'=' * 62}")
    print(f"  BEST MODEL: {best_name}  (Accuracy {best_acc:.4f})")
    print(f"{'=' * 62}")

    # ---- Save ----
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    print(f"\n[OK] Model saved to {MODEL_PATH}")

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"[OK] Vectorizer saved to {VECTORIZER_PATH}")

    # Save accuracy for the API stats endpoint
    acc_path = os.path.join(BASE_DIR, "model_accuracy.txt")
    with open(acc_path, "w") as f:
        f.write(str(round(best_acc * 100, 2)))
    print(f"[OK] Accuracy saved to {acc_path}")

    return best_acc


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 62)
    print("  SPAM DETECTION - Model Training Pipeline")
    print("=" * 62)
    download_dataset()
    accuracy = train_and_compare()
    print(f"\n[OK] Training complete!  Best accuracy: {accuracy:.2%}")
    print("[OK] Ready to serve predictions via app.py")
