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
    GET  /ml-metrics  → model comparison metrics
    GET  /health      → health check

Usage:
    python app.py
"""

import os
import re
import csv
import io
import json
import time
import pickle
import sqlite3
import logging
import argparse
from contextlib import closing
from datetime import datetime, timezone

from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Environment & Logging
# ---------------------------------------------------------------------------
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK — download only if missing
# ---------------------------------------------------------------------------
import nltk

_NLTK_PACKAGES = ["punkt", "punkt_tab", "stopwords", "wordnet"]


def _ensure_nltk_data():
    """Download NLTK data packages only if they are not already present."""
    for pkg in _NLTK_PACKAGES:
        try:
            nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
        except LookupError:
            logger.info("Downloading NLTK package: %s", pkg)
            nltk.download(pkg, quiet=True)


_ensure_nltk_data()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------
# Cached NLP objects (created once, reused for every request)
# ---------------------------------------------------------------------------
_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))

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

# Allow overriding DB path via app.config["DB_PATH"] (useful for tests)
# Flask config values are typically set before the app starts.
try:
    if app.config.get("DB_PATH"):
        DB_PATH = app.config["DB_PATH"]
except Exception:
    pass


FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"
FLASK_HOST = os.environ.get("FLASK_HOST", "127.0.0.1")
FLASK_PORT = int(os.environ.get("FLASK_PORT", "5000"))
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")
OPEN_BROWSER = os.environ.get("OPEN_BROWSER", "1") == "1"

_start_time = time.time()

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
app.config["SECRET_KEY"] = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB max request size

CORS(app, resources={r"/*": {"origins": CORS_ORIGINS}})


# ---------------------------------------------------------------------------
# Security Headers
# ---------------------------------------------------------------------------
@app.after_request
def add_security_headers(response):
    """Add security headers to every response."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # CSP: allow CDN scripts (Three.js, Chart.js) and Google Fonts
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' https://cdnjs.cloudflare.com https://cdn.jsdelivr.net 'unsafe-inline'; "
        "style-src 'self' https://fonts.googleapis.com 'unsafe-inline'; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self' http://127.0.0.1:* http://localhost:*;"
    )
    return response


# ---------------------------------------------------------------------------
# Global Error Handler
# ---------------------------------------------------------------------------
@app.errorhandler(Exception)
def handle_exception(exc):
    """Return JSON for all unhandled errors instead of HTML tracebacks."""
    logger.exception("Unhandled exception: %s", exc)
    code = getattr(exc, "code", 500)
    return jsonify({"error": "Internal server error.", "detail": str(exc)}), code


@app.errorhandler(413)
def handle_too_large(exc):
    return jsonify({"error": "Request payload too large (max 1 MB)."}), 413


@app.errorhandler(404)
def handle_not_found(exc):
    return jsonify({"error": "Resource not found."}), 404


# ---------------------------------------------------------------------------
# Load ML Model
# ---------------------------------------------------------------------------
model = None
vectorizer = None
model_accuracy = 97.8  # default fallback (0-100 scale)


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
        logger.info("Model and vectorizer loaded successfully")
    except FileNotFoundError:
        logger.warning("Model files not found — run 'python train_model.py' first")
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)


load_model()

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def get_db():
    """Get a connection to the SQLite database with WAL mode and busy timeout."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
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

INDEX_SCHEMA = """
    CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
    ON predictions (timestamp DESC)
"""


def init_db():
    """Create the predictions table and indexes if they don't exist."""
    if os.path.exists(DB_PATH) and os.path.getsize(DB_PATH) < 100:
        logger.warning("Corrupted database detected — recreating")
        os.remove(DB_PATH)

    try:
        with closing(get_db()) as conn:
            conn.execute(PREDICTIONS_SCHEMA)
            conn.execute(INDEX_SCHEMA)
            conn.commit()
    except sqlite3.DatabaseError:
        logger.warning("Database unreadable — recreating")
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        with closing(get_db()) as conn:
            conn.execute(PREDICTIONS_SCHEMA)
            conn.execute(INDEX_SCHEMA)
            conn.commit()


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
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [
        _lemmatizer.lemmatize(t)
        for t in tokens
        if t not in _stop_words and len(t) > 1
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
    """Health check with model status and database connectivity."""
    db_ok = False
    try:
        with closing(get_db()) as conn:
            conn.execute("SELECT 1").fetchone()
            db_ok = True
    except Exception:
        pass

    uptime = round(time.time() - _start_time, 1)

    return jsonify({
        "status": "ok",
        "model_loaded": model is not None and vectorizer is not None,
        "database_connected": db_ok,
        "uptime_seconds": uptime,
        "model_accuracy": model_accuracy,
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
    elif hasattr(model, "decision_function"):
        # Fallback for models without predict_proba (e.g. raw LinearSVC)
        import numpy as np
        decision = model.decision_function(features)[0]
        # Sigmoid approximation for confidence
        confidence = float(1 / (1 + np.exp(-abs(decision))))
    else:
        confidence = 0.95

    label = "spam" if prediction == 1 else "ham"
    keywords = detect_keywords(message)
    confidence_pct = round(confidence * 100, 2)

    # Generate AI explanation
    if label == "spam":
        if keywords:
            reason = f"Contains known spam indicators: {', '.join(keywords)}."
            if confidence_pct > 90:
                reason += " High probability based on strong monetary or urgent triggers."
        else:
            reason = "Matches complex spam patterns based on TF-IDF feature weights, despite missing explicit keywords."
    else:
        reason = "Safe language. No significant spam indicators detected in the message structure."

    # Persist to database (store as 0-100 scale)
    with closing(get_db()) as conn:
        conn.execute(
            "INSERT INTO predictions (message, label, confidence, keywords) VALUES (?, ?, ?, ?)",
            (message, label, confidence_pct, ",".join(keywords)),
        )
        conn.commit()

    return jsonify({
        "label": label,
        "confidence": confidence_pct,
        "keywords": keywords,
        "reason": reason,
        "message": message,
    })


@app.route("/history", methods=["GET"])
def get_history():
    """Return all past predictions, newest first."""
    with closing(get_db()) as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY timestamp DESC"
        ).fetchall()

    return jsonify([
        {
            "id": r["id"],
            "message": r["message"],
            "label": r["label"],
            "confidence": round(float(r["confidence"]), 2),
            "keywords": r["keywords"].split(",") if r["keywords"] else [],
            "timestamp": r["timestamp"],
        }
        for r in rows
    ])


@app.route("/history", methods=["DELETE"])
def clear_history():
    """Delete all prediction history."""
    with closing(get_db()) as conn:
        conn.execute("DELETE FROM predictions")
        conn.commit()
    return jsonify({"message": "History cleared successfully."})


@app.route("/stats", methods=["GET"])
def get_stats():
    """Return aggregate prediction statistics."""
    with closing(get_db()) as conn:
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        spam = conn.execute("SELECT COUNT(*) FROM predictions WHERE label='spam'").fetchone()[0]
        ham = conn.execute("SELECT COUNT(*) FROM predictions WHERE label='ham'").fetchone()[0]

    return jsonify({
        "total": total,
        "spam": spam,
        "ham": ham,
        "accuracy": model_accuracy,
    })


@app.route("/export", methods=["GET"])
def export_history():
    """Download prediction history as a CSV file."""
    with closing(get_db()) as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY timestamp DESC"
        ).fetchall()

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


# ---------------------------------------------------------------------------
# Routes — Analytics Dashboard
# ---------------------------------------------------------------------------
@app.route("/api/dashboard/stats", methods=["GET"])
def api_dashboard_stats():
    """Live stats for the dashboard."""
    with closing(get_db()) as conn:
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        spam = conn.execute("SELECT COUNT(*) FROM predictions WHERE label='spam'").fetchone()[0]
        ham = conn.execute("SELECT COUNT(*) FROM predictions WHERE label='ham'").fetchone()[0]
    return jsonify({
        "total_predictions": total,
        "spam_predictions": spam,
        "ham_predictions": ham,
        "accuracy": model_accuracy
    })


@app.route("/api/dashboard/trends", methods=["GET"])
def api_dashboard_trends():
    """Daily prediction trends."""
    with closing(get_db()) as conn:
        rows = conn.execute('''
            SELECT date(timestamp) as date,
                   SUM(CASE WHEN label='spam' THEN 1 ELSE 0 END) as spam_count,
                   SUM(CASE WHEN label='ham' THEN 1 ELSE 0 END) as ham_count
            FROM predictions
            GROUP BY date(timestamp)
            ORDER BY date(timestamp) ASC
            LIMIT 30
        ''').fetchall()
    
    dates = []
    spam_counts = []
    ham_counts = []
    for r in rows:
        dates.append(r["date"])
        spam_counts.append(r["spam_count"])
        ham_counts.append(r["ham_count"])
        
    return jsonify({
        "dates": dates,
        "spam": spam_counts,
        "ham": ham_counts
    })


@app.route("/api/dashboard/confidence", methods=["GET"])
def api_dashboard_confidence():
    """Confidence score distribution (buckets)."""
    with closing(get_db()) as conn:
        rows = conn.execute("SELECT confidence, label FROM predictions").fetchall()
    
    # Create 5 bins: 50-60, 60-70, 70-80, 80-90, 90-100
    bins = {"50-60": 0, "60-70": 0, "70-80": 0, "80-90": 0, "90-100": 0}
    for r in rows:
        conf = float(r["confidence"])
        if conf < 60: bins["50-60"] += 1
        elif conf < 70: bins["60-70"] += 1
        elif conf < 80: bins["70-80"] += 1
        elif conf < 90: bins["80-90"] += 1
        else: bins["90-100"] += 1
        
    return jsonify({"distribution": bins})


@app.route("/api/dashboard/keywords", methods=["GET"])
def api_dashboard_keywords():
    """Top spam keywords from historical data."""
    with closing(get_db()) as conn:
        rows = conn.execute("SELECT keywords FROM predictions WHERE label='spam' AND keywords != ''").fetchall()
    
    freq = {}
    for r in rows:
        for kw in r["keywords"].split(","):
            kw = kw.strip()
            if kw:
                freq[kw] = freq.get(kw, 0) + 1
                
    # Sort by freq
    sorted_kws = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
    return jsonify({"top_keywords": [{"keyword": k, "count": v} for k, v in sorted_kws]})


@app.route("/api/dashboard/history", methods=["GET"])
def api_dashboard_history():
    """Recent 50 history entries for live feed."""
    with closing(get_db()) as conn:
        rows = conn.execute("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 50").fetchall()
    
    return jsonify([
        {
            "id": r["id"],
            "message": r["message"],
            "label": r["label"],
            "confidence": round(float(r["confidence"]), 2),
            "timestamp": r["timestamp"]
        } for r in rows
    ])


@app.route("/api/dashboard/accuracy", methods=["GET"])
def api_dashboard_accuracy():
    """Live accuracy metrics."""
    return jsonify({"accuracy": model_accuracy})


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
    parser = argparse.ArgumentParser(description="SpamDetector API Server")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    args = parser.parse_args()

    url = f"http://{FLASK_HOST}:{FLASK_PORT}"
    print()
    print("=" * 52)
    print("   SPAM DETECTION API SERVER")
    print(f"   {url}")
    print("=" * 52)
    print()

    if OPEN_BROWSER and not args.no_browser:
        import webbrowser
        webbrowser.open(url)

    if FLASK_DEBUG:
        # Development: use Flask's built-in server
        app.run(debug=True, host=FLASK_HOST, port=FLASK_PORT, use_reloader=False)
    else:
        # Production: use Waitress (cross-platform WSGI server)
        try:
            from waitress import serve as waitress_serve
            logger.info("Starting production server (Waitress) on %s", url)
            waitress_serve(app, host=FLASK_HOST, port=FLASK_PORT)
        except ImportError:
            logger.warning("Waitress not installed — falling back to Flask dev server")
            logger.warning("Install waitress for production: pip install waitress")
            app.run(debug=False, host=FLASK_HOST, port=FLASK_PORT, use_reloader=False)
