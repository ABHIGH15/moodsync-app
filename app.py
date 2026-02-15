# app.py
import os, random, logging
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

import csv
from datetime import datetime

LOG_FILE = "logs/user_interactions.csv"

os.makedirs("logs", exist_ok=True)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "mood", "language", "song", "artist"])
# --- Utility imports ---
from utils.preprocessing import load_dataset
from utils.recommend import recommend_songs
from utils.youtube_spotify import attach_links
from utils.face_emotion import detect_face_mood
from utils.text_voice import analyze_text_mood
from utils.mood_mapper import sanitize_mood

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ---- Flask Setup ----
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ---- Supported Lists ----
SUPPORTED_MOODS = ["Calm", "Happy", "Sad", "Energetic"]
SUPPORTED_LANGUAGES = ["English", "Hindi", "Tamil", "Marathi", "Punjabi"]

# ---- Mood → Theme Map (emoji + Tailwind color) ----
MOOD_THEME = {
    "Calm":      {"emoji": "🧘", "color": "emerald", "bg": "bg-emerald-500/20"},
    "Happy":     {"emoji": "😊", "color": "yellow",  "bg": "bg-yellow-500/20"},
    "Sad":       {"emoji": "😔", "color": "blue",    "bg": "bg-blue-500/20"},
    "Energetic": {"emoji": "⚡", "color": "pink",    "bg": "bg-pink-500/20"},
}

# ---- File Paths ----
DATA_PATH = "data/final_master_song_dataset.csv"
MODEL_PATH = "model/final_mood_classifier_v1.joblib"

# ---- Model & Data Loading ----
logging.info("🔄 Loading dataset and model...")
try:
    df = load_dataset(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    logging.info(f"✅ Dataset loaded: {df.shape[0]} rows | Model OK")
except Exception as e:
    logging.error(f"⚠️ Could not load model or dataset: {e}")
    df, model = pd.DataFrame(), None


# ---- Home Route ----
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        language = (request.form.get("language") or "").strip()
        text_input = (request.form.get("feeling_text") or "").strip()
        image = request.files.get("face")

        mood_input = None

        # 1️⃣ Face detection (stub → Calm)
        if image and image.filename:
            try:
                image_path = os.path.join(app.config["UPLOAD_FOLDER"], image.filename)
                image.save(image_path)
                mood_input = detect_face_mood(image_path)
                os.remove(image_path)
            except Exception as e:
                logging.warning(f"⚠️ Face detection failed: {e}")

        # 2️⃣ Text mood detection
        if not mood_input and text_input:
            try:
                mood_input = analyze_text_mood(text_input)
            except Exception as e:
                logging.warning(f"⚠️ Text mood analysis failed: {e}")

        # 3️⃣ Manual fallback
        if not mood_input:
            mood_input = (request.form.get("mood") or "Calm").strip()

        mood = sanitize_mood(mood_input)
        if mood not in SUPPORTED_MOODS:
            mood = "Calm"
        if language and language not in SUPPORTED_LANGUAGES:
            language = ""

        logging.info(f"🎭 Final Mood: {mood} | Language: {language or 'Any'}")

        # Generate recommendations
        try:
            rec_df = recommend_songs(df, mood, language)
            songs = rec_df.to_dict(orient="records") if not rec_df.empty else []
            songs = attach_links(songs)
            random.shuffle(songs)

            # Split: top 5 + remainder
            top_mix, rest = songs[:5], songs[5:]
        except Exception as e:
            logging.error(f"⚠️ Recommendation generation error: {e}")
            top_mix, rest = [], []

        theme = MOOD_THEME.get(mood, MOOD_THEME["Calm"])

        return render_template(
            "results.html",
            mood=mood,
            language=language or "All",
            top_mix=top_mix,
            songs=rest,
            moods=SUPPORTED_MOODS,
            languages=SUPPORTED_LANGUAGES,
            theme=theme
        )

    # Default: GET → main form
    return render_template("index.html", moods=SUPPORTED_MOODS, languages=SUPPORTED_LANGUAGES, theme=MOOD_THEME["Calm"])


# ---- JSON API ----
@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    data = request.json or {}
    mood = sanitize_mood(data.get("mood", "Calm"))
    language = data.get("language", "")

    if mood not in SUPPORTED_MOODS: mood = "Calm"
    if language not in SUPPORTED_LANGUAGES: language = ""

    try:
        rec_df = recommend_songs(df, mood, language)
        songs = rec_df.to_dict(orient="records") if not rec_df.empty else []
        random.shuffle(songs)
        return jsonify(attach_links(songs))
    except Exception as e:
        logging.error(f"⚠️ API recommendation error: {e}")
        return jsonify({"error": "Recommendation generation failed"}), 500


# ---- Health Check ----
@app.route("/ping")
def ping():
    return {
        "status": "ok",
        "message": "MoodSync backend running successfully 🎶",
        "moods": SUPPORTED_MOODS,
        "languages": SUPPORTED_LANGUAGES,
        "dataset_rows": int(df.shape[0]) if isinstance(df, pd.DataFrame) else 0,
        "team": "Predix"
    }
@app.route("/log_click", methods=["POST"])
def log_click():
    data = request.json or {}

    mood = data.get("mood", "")
    language = data.get("language", "")
    song = data.get("song", "")
    artist = data.get("artist", "")

    try:
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat(),
                mood,
                language,
                song,
                artist
            ])
        return {"status": "logged"}
    except Exception as e:
        logging.error(f"Logging error: {e}")
        return {"status": "error"}, 500

# ---- Entry Point ----
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logging.info(f"🚀 Starting MoodSync on port {port}")
    app.run(host="0.0.0.0", port=port)
