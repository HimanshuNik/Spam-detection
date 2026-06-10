"""
Spam Detection — Flask API Server
===================================
REST API for spam/ham prediction with SQLite-backed history.
Also serves the frontend as static files.

Endpoints:
    GET  /            → serve frontend
    POST /predict     → classify a message
    GET  /history     → list past predictions
    DELETE /history   → clear history
    GET  /stats       → aggregate statistics
    GET  /export      → download history as CSV
    GET  /samples     → sample spam & ham messages

Usage:
    python app.py
"""

import os
import re
import csv
import io
import json
import pickle
import sqlite3
from datetime import datetime


import webbrowser

from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS

# --- NLTK (must match training preprocessing) ---
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
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
ACCURACY_PATH = os.path.join(BASE_DIR, "model_accuracy.txt")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")
DB_PATH = os.path.join(BASE_DIR, "spam_history.db")

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app, resources={r"/*": {"origins": "*"}})

# ---------------------------------------------------------------------------
# Load ML Model
# ---------------------------------------------------------------------------
model = None
vectorizer = None
model_accuracy = 97.8  # default fallback


def load_model():
    """Load the trained model and vectorizer from disk."""
    global model, vectorizer, model_accuracy
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        if os.path.exists(ACCURACY_PATH):
            with open(ACCURACY_PATH, "r") as f:
                model_accuracy = float(f.read().strip())
        print("[OK] Model and vectorizer loaded")
    except FileNotFoundError:
        print("[!] Model files not found - run  python train_model.py  first")
    except Exception as exc:
        print(f"[!] Failed to load model: {exc}")


load_model()

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def get_db():
    """Get a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


PREDICTIONS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS predictions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        message     TEXT    NOT NULL,
        label       TEXT    NOT NULL,
        confidence  REAL    NOT NULL,
        keywords    TEXT,
        timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
    )
"""


def init_db():
    """Create the predictions table if it doesn't exist."""
    if os.path.exists(DB_PATH) and os.path.getsize(DB_PATH) < 100:
        print("[!] Corrupted database detected — recreating …")
        os.remove(DB_PATH)

    try:
        conn = get_db()
        conn.execute(PREDICTIONS_SCHEMA)
        conn.commit()
        conn.close()
    except sqlite3.DatabaseError:
        print("[!] Database unreadable — recreating …")
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        conn = get_db()
        conn.execute(PREDICTIONS_SCHEMA)
        conn.commit()
        conn.close()


init_db()


def parse_json_request():
    """Parse JSON from the request body, even when the Content-Type header is missing."""
    if request.is_json:
        return request.get_json(silent=True)

    try:
        raw = request.data.decode("utf-8") if request.data else ""
        return json.loads(raw) if raw else None
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# NLP Preprocessing  (must mirror train_model.py exactly)
# ---------------------------------------------------------------------------
SPAM_KEYWORDS = [
    "free", "win", "winner", "cash", "prize", "claim", "urgent",
    "congratulations", "offer", "deal", "discount", "limited", "act now",
    "click", "subscribe", "credit", "loan", "income", "earn", "money",
    "buy", "order", "purchase", "cheap", "bargain", "bonus", "guaranteed",
    "no cost", "risk free", "call now", "apply", "selected", "exclusive",
    "promotion", "reward", "txt", "text", "mobile", "phone", "ringtone",
    "account", "verify", "confirm", "password", "bank", "paypal",
    "viagra", "pharmacy", "weight loss", "diet", "supplement",
    "100%", "act now", "click here", "congratulation", "expire",
]


def preprocess_text(text: str) -> str:
    """NLP preprocessing identical to the training pipeline."""
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


def detect_keywords(text: str) -> list:
    """Find spam-indicator keywords present in the original message."""
    text_lower = text.lower()
    return [kw for kw in SPAM_KEYWORDS if kw in text_lower]


# ---------------------------------------------------------------------------
# Routes — API (registered before static catch-all)
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    """Lightweight health check for frontend connectivity probing."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None and vectorizer is not None,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Classify a message as spam or ham.

    Request body:  { "message": "..." }
    Response:      { "label", "confidence", "keywords", "message" }
    """
    if model is None or vectorizer is None:
        load_model()
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    data = parse_json_request()
    if data is None:
        return jsonify({"error": "Invalid JSON in request body."}), 400
    if "message" not in data:
        return jsonify({"error": "Missing 'message' field in request body."}), 400

    message = data["message"].strip()
    if not message:
        return jsonify({"error": "Message cannot be empty."}), 400
    if len(message) > 10_000:
        return jsonify({"error": "Message too long (max 10 000 chars)."}), 400

    # Predict
    processed = preprocess_text(message)
    features = vectorizer.transform([processed])
    prediction = model.predict(features)[0]

    # Confidence score
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        confidence = float(max(probs))
    else:
        confidence = 0.95

    label = "spam" if prediction == 1 else "ham"
    keywords = detect_keywords(message)

    # Persist to database
    conn = get_db()
    conn.execute(
        "INSERT INTO predictions (message, label, confidence, keywords) VALUES (?, ?, ?, ?)",
        (message, label, confidence, ",".join(keywords)),
    )
    conn.commit()
    conn.close()

    return jsonify({
        "label": label,
        "confidence": round(confidence * 100, 2),
        "keywords": keywords,
        "message": message,
    })


@app.route("/history", methods=["GET"])
def get_history():
    """Return all past predictions, newest first."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY timestamp DESC"
    ).fetchall()
    conn.close()

    return jsonify([
        {
            "id": r["id"],
            "message": r["message"],
            "label": r["label"],
            "confidence": round(float(r["confidence"]) * 100, 2)
            if float(r["confidence"]) <= 1
            else float(r["confidence"]),
            "keywords": r["keywords"].split(",") if r["keywords"] else [],
            "timestamp": r["timestamp"],
        }
        for r in rows
    ])


@app.route("/history", methods=["DELETE"])
def clear_history():
    """Delete all prediction history."""
    conn = get_db()
    conn.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
    return jsonify({"message": "History cleared successfully."})


@app.route("/stats", methods=["GET"])
def get_stats():
    """Return aggregate prediction statistics."""
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    spam  = conn.execute("SELECT COUNT(*) FROM predictions WHERE label='spam'").fetchone()[0]
    ham   = conn.execute("SELECT COUNT(*) FROM predictions WHERE label='ham'").fetchone()[0]
    conn.close()

    return jsonify({
        "total": total,
        "spam": spam,
        "ham": ham,
        "accuracy": model_accuracy,
    })


@app.route("/export", methods=["GET"])
def export_history():
    """Download prediction history as a CSV file."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY timestamp DESC"
    ).fetchall()
    conn.close()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["ID", "Message", "Label", "Confidence", "Keywords", "Timestamp"])
    for r in rows:
        writer.writerow([r["id"], r["message"], r["label"],
                         r["confidence"], r["keywords"], r["timestamp"]])

    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=spam_history.csv"},
    )


@app.route("/ml-metrics", methods=["GET"])
def get_ml_metrics():
    """Return model comparison metrics for the ML analytics dashboard."""
    if not os.path.exists(METRICS_PATH):
        return jsonify({
            "error": "Metrics not found. Run python train_model.py first.",
        }), 404

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.route("/samples", methods=["GET"])
def get_samples():
    """Return example spam & ham messages for quick testing."""
    return jsonify({
        "spam": [
            "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
            "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
            "URGENT! You have won a 1 week free membership in our £100,000 Prize Jackpot! Txt the word: claim to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18",
            "Congratulations! You've been selected for a free iPhone 15 Pro! Click here to claim your prize now: http://spam-link.com/claim",
            "Your account has been compromised! Verify your identity immediately by clicking this link or your account will be suspended within 24 hours.",
        ],
        "ham": [
            "Hey, are you coming to the party tonight? Let me know so I can pick you up.",
            "I'll be late for dinner. Got stuck in traffic. Save some food for me please!",
            "Can you send me the notes from yesterday's meeting? I missed the last part.",
            "Happy birthday! Hope you have an amazing day. Let's catch up this weekend!",
            "Just finished the project report. I'll email it to you by end of day.",
        ],
    })


# ---------------------------------------------------------------------------
# Routes — Frontend
# ---------------------------------------------------------------------------
@app.route("/")
def serve_frontend():
    """Serve the single-page application."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/dashboard")
@app.route("/dashboard.html")
def serve_dashboard():
    """Serve the ML analytics dashboard."""
    return send_from_directory(FRONTEND_DIR, "dashboard.html")


@app.route("/<path:path>")
def serve_static(path):
    """Serve static assets (CSS, JS, images)."""
    return send_from_directory(FRONTEND_DIR, path)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    url = "http://127.0.0.1:5000"
    print()
    print("=" * 52)
    print("   SPAM DETECTION API SERVER")
    print(f"   {url}")
    print("=" * 52)
    print()
    webbrowser.open(url)
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
