import os, logging, csv
import pandas as pd
import joblib
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file

# --- Utility imports ---
from utils.preprocessing import load_dataset
from utils.recommend import recommend_songs
from utils.youtube_spotify import attach_links
from utils.face_emotion import detect_face_mood
from utils.text_voice import analyze_text_mood
from utils.mood_mapper import sanitize_mood

# =========================================================
# Logging
# =========================================================
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# =========================================================
# Flask Setup
# =========================================================
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# =========================================================
# Interaction Log File
# =========================================================
LOG_FILE = "logs/user_interactions.csv"
os.makedirs("logs", exist_ok=True)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "mood", "language", "song", "artist", "platform"])

# =========================================================
# Supported Lists
# =========================================================
SUPPORTED_MOODS = ["Calm", "Happy", "Sad", "Energetic"]
SUPPORTED_LANGUAGES = ["English", "Hindi", "Tamil", "Marathi", "Punjabi"]

MOOD_THEME = {
    "Calm": {"emoji": "🧘", "bg": "bg-emerald-500/20"},
    "Happy": {"emoji": "😊", "bg": "bg-yellow-500/20"},
    "Sad": {"emoji": "😔", "bg": "bg-blue-500/20"},
    "Energetic": {"emoji": "⚡", "bg": "bg-pink-500/20"},
}

# =========================================================
# Paths
# =========================================================
DATA_PATH = "data/final_master_song_dataset.csv"
MODEL_PATH = "model/final_mood_classifier_v1.joblib"

# =========================================================
# Load Dataset + Model
# =========================================================
logging.info("🔄 Loading dataset and model...")
try:
    df = load_dataset(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    logging.info(f"✅ Dataset loaded: {df.shape[0]} rows | Model OK")
except Exception as e:
    logging.error(f"⚠️ Could not load model or dataset: {e}")
    df, model = pd.DataFrame(), None


# =========================================================
# 🔥 CONTEXTUAL ADAPTIVE HYBRID RANKING
# =========================================================
def apply_contextual_hybrid_ranking(songs, current_mood, current_language):

    if not songs:
        return songs

    songs_df = pd.DataFrame(songs)

    # Ensure confidence exists
    if "confidence" not in songs_df.columns:
        songs_df["confidence"] = 0.5  # fallback neutral score

    songs_df["confidence"] = songs_df["confidence"].fillna(0.5)

    # If no logs, return sorted by confidence only
    if not os.path.exists(LOG_FILE):
        return songs_df.sort_values(by="confidence", ascending=False).to_dict(orient="records")

    try:
        logs_df = pd.read_csv(LOG_FILE)

        if logs_df.empty:
            return songs_df.sort_values(by="confidence", ascending=False).to_dict(orient="records")

        # Global click count
        global_counts = logs_df.groupby(["song", "artist"]).size().reset_index(name="global_click")

        # Mood specific click count
        mood_counts = logs_df[logs_df["mood"] == current_mood] \
            .groupby(["song", "artist"]).size().reset_index(name="mood_click")

        # Language specific click count
        language_counts = logs_df[logs_df["language"] == current_language] \
            .groupby(["song", "artist"]).size().reset_index(name="language_click")

        # Merge all
        merged = songs_df.merge(global_counts, how="left",
                                left_on=["name", "artist"],
                                right_on=["song", "artist"])

        merged = merged.merge(mood_counts, how="left",
                              left_on=["name", "artist"],
                              right_on=["song", "artist"],
                              suffixes=("", "_mood"))

        merged = merged.merge(language_counts, how="left",
                              left_on=["name", "artist"],
                              right_on=["song", "artist"],
                              suffixes=("", "_lang"))

        merged["global_click"] = merged["global_click"].fillna(0)
        merged["mood_click"] = merged["mood_click"].fillna(0)
        merged["language_click"] = merged["language_click"].fillna(0)

        # Normalize click scores
        if merged["mood_click"].max() > 0:
            merged["mood_score"] = merged["mood_click"] / merged["mood_click"].max()
        else:
            merged["mood_score"] = 0

        if merged["language_click"].max() > 0:
            merged["language_score"] = merged["language_click"] / merged["language_click"].max()
        else:
            merged["language_score"] = 0

        # Hybrid score formula
        merged["hybrid_score"] = (
            0.6 * merged["confidence"] +
            0.25 * merged["mood_score"] +
            0.15 * merged["language_score"]
        )

        merged = merged.sort_values(by="hybrid_score", ascending=False)

        return merged.to_dict(orient="records")

    except Exception as e:
        logging.error(f"Hybrid ranking failed: {e}")
        return songs_df.sort_values(by="confidence", ascending=False).to_dict(orient="records")


# =========================================================
# HOME ROUTE
# =========================================================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        language = (request.form.get("language") or "").strip()
        text_input = (request.form.get("feeling_text") or "").strip()
        image = request.files.get("face")

        mood_input = None

        if image and image.filename:
            try:
                image_path = os.path.join(app.config["UPLOAD_FOLDER"], image.filename)
                image.save(image_path)
                mood_input = detect_face_mood(image_path)
                os.remove(image_path)
            except Exception as e:
                logging.warning(f"⚠️ Face detection failed: {e}")

        if not mood_input and text_input:
            try:
                mood_input = analyze_text_mood(text_input)
            except Exception as e:
                logging.warning(f"⚠️ Text mood analysis failed: {e}")

        if not mood_input:
            mood_input = (request.form.get("mood") or "Calm").strip()

        mood = sanitize_mood(mood_input)

        if mood not in SUPPORTED_MOODS:
            mood = "Calm"

        if language and language not in SUPPORTED_LANGUAGES:
            language = ""

        logging.info(f"🎭 Final Mood: {mood} | Language: {language or 'Any'}")

        try:
            rec_df = recommend_songs(df, mood, language)
            songs = rec_df.to_dict(orient="records") if not rec_df.empty else []
            songs = attach_links(songs)

            # 🔥 Apply Hybrid Ranking
            songs = apply_contextual_hybrid_ranking(songs, mood, language)

            top_mix = songs[:5]
            rest = songs[5:]

        except Exception as e:
            logging.error(f"⚠️ Recommendation error: {e}")
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

    return render_template(
        "index.html",
        moods=SUPPORTED_MOODS,
        languages=SUPPORTED_LANGUAGES,
        theme=MOOD_THEME["Calm"]
    )


# =========================================================
# CLICK LOGGER
# =========================================================
@app.route("/log_click", methods=["POST"])
def log_click():
    try:
        data = request.json or {}

        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat(),
                data.get("mood", ""),
                data.get("language", ""),
                data.get("song", ""),
                data.get("artist", ""),
                data.get("platform", "")
            ])

        logging.info(f"📝 Logged click: {data.get('song')}")
        return {"status": "logged"}

    except Exception as e:
        logging.error(f"Logging error: {e}")
        return {"status": "error"}, 500


# =========================================================
# DOWNLOAD LOGS
# =========================================================
@app.route("/download_logs")
def download_logs():
    return send_file(LOG_FILE, as_attachment=True)



# =========================================================
# PUBLIC DASHBOARD
# =========================================================
@app.route("/dashboard")
def dashboard():

    try:
        if not os.path.exists(LOG_FILE):
            return render_template("dashboard.html",
                                   total_clicks=0,
                                   top_songs=[],
                                   mood_dist={},
                                   language_dist={},
                                   platform_dist={},
                                   spark_insights={})

        logs_df = pd.read_csv(LOG_FILE)

        if logs_df.empty:
            return render_template("dashboard.html",
                                   total_clicks=0,
                                   top_songs=[],
                                   mood_dist={},
                                   language_dist={},
                                   platform_dist={},
                                   spark_insights={})

        total_clicks = len(logs_df)

        # Top Songs (Global)
        top_songs = (
            logs_df.groupby(["song", "artist"])
            .size()
            .reset_index(name="count")
            .sort_values(by="count", ascending=False)
            .head(5)
            .to_dict(orient="records")
        )

        # Mood distribution
        mood_dist = logs_df["mood"].value_counts().to_dict()

        # Language distribution
        language_dist = logs_df["language"].value_counts().to_dict()

        # Platform distribution
        platform_dist = logs_df["platform"].value_counts().to_dict()

        # Load Spark insights if exists
        spark_file = "data/spark_insights.json"
        if os.path.exists(spark_file):
            import json
            with open(spark_file, "r") as f:
                spark_insights = json.load(f)
        else:
            spark_insights = {}

        return render_template("dashboard.html",
                               total_clicks=total_clicks,
                               top_songs=top_songs,
                               mood_dist=mood_dist,
                               language_dist=language_dist,
                               platform_dist=platform_dist,
                               spark_insights=spark_insights)

    except Exception as e:
        logging.error(f"Dashboard error: {e}")
        return "Dashboard Error", 500
# =========================================================
# HEALTH CHECK
# =========================================================
@app.route("/ping")
def ping():
    return {
        "status": "ok",
        "dataset_rows": int(df.shape[0]) if isinstance(df, pd.DataFrame) else 0,
        "team": "Predix"
    }


# =========================================================
# ENTRY
# =========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logging.info(f"🚀 Starting MoodSync on port {port}")
    app.run(host="0.0.0.0", port=port)
