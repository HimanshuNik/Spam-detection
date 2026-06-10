"""
Spam Detection — Advanced Model Training Pipeline
==================================================
Trains and compares four classifiers (Naive Bayes, Logistic Regression,
Random Forest, SVM) with full NLP preprocessing, evaluation metrics,
confusion matrices, and exports data + charts for the ML dashboard.

Usage:
    python train_model.py
"""

import os
import re
import json
import time
import pickle
import zipfile
import urllib.request
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
)
DATASET_PATH = os.path.join(BASE_DIR, "spam.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
ACCURACY_PATH = os.path.join(BASE_DIR, "model_accuracy.txt")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")
CHARTS_DIR = os.path.join(PROJECT_DIR, "frontend", "charts")

MODEL_COLORS = {
    "Naive Bayes": "#8b5cf6",
    "Logistic Regression": "#3b82f6",
    "Random Forest": "#06b6d4",
    "Support Vector Machine": "#10b981",
}


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

        tsv_path = os.path.join(BASE_DIR, "SMSSpamCollection")
        df = pd.read_csv(
            tsv_path, sep="\t", header=None, names=["label", "message"], encoding="latin-1"
        )
        df.to_csv(DATASET_PATH, index=False)

        for tmp in [zip_path, tsv_path, os.path.join(BASE_DIR, "readme")]:
            if os.path.exists(tmp):
                os.remove(tmp)

        print(f"[OK] Dataset saved — {len(df)} messages")

    except Exception as exc:
        print(f"[!] Download failed: {exc}")
        print("[*] Creating built-in sample dataset ...")
        _create_fallback_dataset()


def _create_fallback_dataset():
    """Create a representative dataset when download is unavailable."""
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
        "Act now! Limited time offer - get 90% , discount on all products. Click here!",
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
    pd.DataFrame(data).to_csv(DATASET_PATH, index=False)
    print(f"[OK] Fallback dataset created — {len(data)} messages")


# ---------------------------------------------------------------------------
# NLP Preprocessing
# ---------------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline:
      1. Lowercasing
      2. Remove URLs, numbers, punctuation
      3. Tokenization
      4. Stopword removal
      5. Lemmatization
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words and len(t) > 1
    ]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Model explanation
# ---------------------------------------------------------------------------
def build_model_explanation(best_name: str, results: list) -> str:
    """Generate a human-readable explanation for the selected production model."""
    best = next(r for r in results if r["name"] == best_name)
    others = [r for r in results if r["name"] != best_name]

    avg_other_acc = np.mean([r["accuracy"] for r in others])
    avg_other_f1 = np.mean([r["f1"] for r in others])

    explanations = {
        "Naive Bayes": (
            "Multinomial Naive Bayes excels on high-dimensional sparse TF-IDF text features. "
            "It treats word occurrences as conditionally independent, which is a strong assumption "
            "for bag-of-words spam detection and trains extremely fast on large vocabularies."
        ),
        "Logistic Regression": (
            "Logistic Regression provides strong linear decision boundaries in TF-IDF space with "
            "well-calibrated probabilities. It balances interpretability and performance but can "
            "be sensitive to class imbalance without careful tuning."
        ),
        "Random Forest": (
            "Random Forest ensembles many decision trees to capture non-linear patterns. "
            "While powerful on structured data, it is slower on sparse text matrices and may "
            "overfit high-dimensional TF-IDF features compared to linear models."
        ),
        "Support Vector Machine": (
            "Linear SVM finds an optimal separating hyperplane with maximum margin in TF-IDF space. "
            "It is robust to high dimensionality but lacks native probability estimates and is "
            "typically slower to train than Naive Bayes on very large vocabularies."
        ),
    }

    base = explanations.get(best_name, f"{best_name} achieved the highest accuracy on the test set.")

    return (
        f"{base} "
        f"On the held-out test set, {best_name} reached {best['accuracy']:.1%} accuracy "
        f"and {best['f1']:.1%} F1-score — "
        f"{(best['accuracy'] - avg_other_acc) * 100:+.1f} percentage points above the average "
        f"accuracy of the other models ({avg_other_acc:.1%}) and "
        f"{(best['f1'] - avg_other_f1) * 100:+.1f} points above their average F1 ({avg_other_f1:.1%}). "
        f"Combined with TF-IDF preprocessing (lowercasing, stopword removal, tokenization, "
        f"lemmatization), it was selected as the production model for real-time spam detection."
    )


# ---------------------------------------------------------------------------
# Chart generation (Matplotlib + Seaborn)
# ---------------------------------------------------------------------------
def _chart_style():
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update({
        "figure.facecolor": "#0c0c1f",
        "axes.facecolor": "#141432",
        "axes.edgecolor": "#333355",
        "axes.labelcolor": "#f0f0f5",
        "text.color": "#f0f0f5",
        "xtick.color": "#f0f0f5",
        "ytick.color": "#f0f0f5",
        "grid.color": "#333355",
        "legend.facecolor": "#141432",
        "legend.edgecolor": "#333355",
    })


def save_charts(results: list, label_counts: dict):
    """Export static comparison charts for the dashboard."""
    os.makedirs(CHARTS_DIR, exist_ok=True)
    _chart_style()

    names = [r["name"] for r in results]
    colors = [MODEL_COLORS.get(n, "#8b5cf6") for n in names]

    # Accuracy bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    accs = [r["accuracy"] * 100 for r in results]
    bars = ax.bar(names, accs, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Accuracy Comparison", fontweight="bold", pad=16)
    ax.set_ylim(0, 105)
    plt.xticks(rotation=15, ha="right")
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "accuracy_comparison.png"), dpi=150)
    plt.close(fig)

    # Precision / Recall / F1 grouped bar chart
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(names))
    width = 0.25
    metrics = ["precision", "recall", "f1"]
    metric_colors = ["#8b5cf6", "#3b82f6", "#10b981"]
    for i, (metric, color) in enumerate(zip(metrics, metric_colors)):
        vals = [r[metric] * 100 for r in results]
        ax.bar(x + (i - 1) * width, vals, width, label=metric.capitalize(), color=color, alpha=0.9)
    ax.set_ylabel("Score (%)")
    ax.set_title("Precision, Recall & F1 Comparison", fontweight="bold", pad=16)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "metrics_comparison.png"), dpi=150)
    plt.close(fig)

    # Spam vs Ham pie chart
    fig, ax = plt.subplots(figsize=(6, 6))
    labels_pie = ["Ham (Legitimate)", "Spam"]
    sizes = [label_counts.get("ham", 0), label_counts.get("spam", 0)]
    pie_colors = ["#10b981", "#ef4444"]
    ax.pie(sizes, labels=labels_pie, autopct="%1.1f%%", colors=pie_colors,
           startangle=90, textprops={"color": "#f0f0f5"})
    ax.set_title("Dataset: Spam vs Ham Distribution", fontweight="bold", pad=16)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "spam_ham_distribution.png"), dpi=150)
    plt.close(fig)

    # Confusion matrix per model
    for r in results:
        cm = np.array(r["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Pred Ham", "Pred Spam"],
            yticklabels=["Actual Ham", "Actual Spam"],
            ax=ax, linewidths=0.5, linecolor="#333355",
        )
        ax.set_title(f"Confusion Matrix — {r['name']}", fontweight="bold", pad=12)
        fig.tight_layout()
        slug = r["id"]
        fig.savefig(os.path.join(CHARTS_DIR, f"confusion_{slug}.png"), dpi=150)
        plt.close(fig)

    print(f"[OK] Charts saved to {CHARTS_DIR}")


# ---------------------------------------------------------------------------
# Training & Comparison
# ---------------------------------------------------------------------------
def train_and_compare():
    """Train all four models, evaluate, save best model + dashboard metrics."""

    print("\n[*] Loading dataset ...")
    df = pd.read_csv(DATASET_PATH, encoding="latin-1")

    if "label" not in df.columns:
        df.columns = ["label", "message"] + list(df.columns[2:])
    df = df[["label", "message"]].dropna()
    df["label"] = df["label"].str.lower().str.strip()

    label_counts = df["label"].value_counts().to_dict()
    print(f"[OK] {len(df)} messages  |  {label_counts}")

    print("[*] Preprocessing text (lowercase → tokenize → stopwords → lemmatize) ...")
    df["processed"] = df["message"].apply(preprocess_text)
    df["label_enc"] = (df["label"] == "spam").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df["processed"],
        df["label_enc"],
        test_size=0.2,
        random_state=42,
        stratify=df["label_enc"],
    )

    print("[*] Building TF-IDF features (max 5 000, unigrams + bigrams) ...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model_defs = {
        "naive_bayes": ("Naive Bayes", MultinomialNB(alpha=0.1)),
        "logistic_regression": (
            "Logistic Regression",
            LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        ),
        "random_forest": (
            "Random Forest",
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        ),
        "svm": (
            "Support Vector Machine",
            LinearSVC(C=1.0, random_state=42, max_iter=3000),
        ),
    }

    results = []
    best_model = None
    best_acc = 0.0
    best_name = ""

    print("\n" + "=" * 62)
    print("  MODEL COMPARISON RESULTS")
    print("=" * 62)

    for model_id, (name, clf) in model_defs.items():
        t0 = time.perf_counter()
        clf.fit(X_train_vec, y_train)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        preds = clf.predict(X_test_vec)
        predict_time = time.perf_counter() - t1

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        cm = confusion_matrix(y_test, preds).tolist()

        result = {
            "id": model_id,
            "name": name,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "train_time_sec": round(train_time, 4),
            "predict_time_sec": round(predict_time, 4),
            "total_time_sec": round(train_time + predict_time, 4),
            "confusion_matrix": cm,
            "color": MODEL_COLORS.get(name, "#8b5cf6"),
        }
        results.append(result)

        print(f"\n  {name}")
        print(f"    Accuracy : {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall   : {rec:.4f}")
        print(f"    F1-Score : {f1:.4f}")
        print(f"    Train    : {train_time:.4f}s  |  Predict: {predict_time:.4f}s")

        if acc > best_acc:
            best_acc = acc
            best_model = clf
            best_name = name

    best_accuracy = max(results, key=lambda r: r["accuracy"])
    fastest = min(results, key=lambda r: r["total_time_sec"])
    highest_f1 = max(results, key=lambda r: r["f1"])

    explanation = build_model_explanation(best_name, results)

    print(f"\n{'=' * 62}")
    print(f"  BEST MODEL (production): {best_name}  (Accuracy {best_acc:.4f})")
    print(f"  FASTEST MODEL          : {fastest['name']}  ({fastest['total_time_sec']:.4f}s)")
    print(f"  HIGHEST F1             : {highest_f1['name']}  (F1 {highest_f1['f1']:.4f})")
    print(f"{'=' * 62}")

    metrics_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "preprocessing": [
            "Lowercasing",
            "URL & number removal",
            "Tokenization",
            "Stopword removal",
            "Lemmatization",
            "TF-IDF vectorization (5000 features, 1–2 grams)",
        ],
        "dataset": {
            "total": len(df),
            "spam": int(label_counts.get("spam", 0)),
            "ham": int(label_counts.get("ham", 0)),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "test_split": 0.2,
        },
        "models": results,
        "highlights": {
            "best_accuracy": {
                "model": best_accuracy["name"],
                "model_id": best_accuracy["id"],
                "value": best_accuracy["accuracy"],
            },
            "fastest": {
                "model": fastest["name"],
                "model_id": fastest["id"],
                "value": fastest["total_time_sec"],
                "unit": "seconds",
            },
            "highest_f1": {
                "model": highest_f1["name"],
                "model_id": highest_f1["id"],
                "value": highest_f1["f1"],
            },
        },
        "selected_model": {
            "name": best_name,
            "accuracy": round(best_acc, 4),
            "reason": explanation,
        },
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    print(f"\n[OK] Model saved to {MODEL_PATH}")

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"[OK] Vectorizer saved to {VECTORIZER_PATH}")

    with open(ACCURACY_PATH, "w") as f:
        f.write(str(round(best_acc * 100, 2)))
    print(f"[OK] Accuracy saved to {ACCURACY_PATH}")

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[OK] Metrics JSON saved to {METRICS_PATH}")

    save_charts(results, label_counts)

    return best_acc, metrics_payload


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 62)
    print("  SPAM DETECTION — Advanced ML Training Pipeline")
    print("=" * 62)
    download_dataset()
    accuracy, _ = train_and_compare()
    print(f"\n[OK] Training complete!  Best accuracy: {accuracy:.2%}")
    print("[OK] Open /dashboard.html for the ML analytics dashboard")
