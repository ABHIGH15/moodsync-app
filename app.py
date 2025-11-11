# app.py
import os
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

# --- Utility imports ---
from utils.preprocessing import load_dataset
from utils.recommend import recommend_songs
from utils.youtube_spotify import attach_links
from utils.face_emotion import detect_face_mood
from utils.text_voice import analyze_text_mood
from utils.mood_mapper import sanitize_mood

import logging
logging.basicConfig(level=logging.DEBUG)

# --- Flask Setup ---
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# --- Paths ---
DATA_PATH = "data/final_master_song_dataset.csv"
MODEL_PATH = "model/final_mood_classifier_v1.joblib"

# --- Load Model & Dataset ---
print("🔄 Loading data and model...")
try:
    df = load_dataset(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    print("✅ Model and dataset loaded successfully.")
except Exception as e:
    print(f"⚠️ Error loading model or dataset: {e}")
    df = pd.DataFrame()
    model = None

# --- Home Route ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        mood_input = None
        language = request.form.get("language", "").strip()
        text_input = request.form.get("user_text", "").strip()
        image = request.files.get("face_image")

        # 1️⃣ Face Detection
        if image and image.filename != "":
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], image.filename)
            image.save(image_path)
            mood_input = detect_face_mood(image_path)
            os.remove(image_path)

        # 2️⃣ Text Input
        elif text_input:
            mood_input = analyze_text_mood(text_input)

        # 3️⃣ Default
        else:
            mood_input = "Calm"

        mood = sanitize_mood(mood_input)
        print(f"🎭 Mood detected: {mood} | Language: {language or 'Any'}")

        try:
            recommendations = recommend_songs(df, mood, language)
            songs_dict = recommendations.to_dict(orient="records")
            songs_with_links = attach_links(songs_dict)
        except Exception as e:
            print(f"⚠️ Recommendation error: {e}")
            songs_with_links = []

        return render_template("results.html",
                               mood=mood,
                               language=language or "All",
                               songs=songs_with_links)

    return render_template("index.html")

# --- API Endpoint ---
@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    data = request.json or {}
    mood = sanitize_mood(data.get("mood", "Calm"))
    language = data.get("language", "")
    recommendations = recommend_songs(df, mood, language)
    songs_dict = recommendations.to_dict(orient="records")
    return jsonify(attach_links(songs_dict))

# --- Health Check ---
@app.route("/ping")
def ping():
    return {"status": "ok", "message": "MoodSync backend running successfully 🎶"}

# --- App Entry Point ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

