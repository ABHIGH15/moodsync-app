# app.py
import os
from flask import Flask, render_template, request
import pandas as pd
import joblib

# Import utility functions
from utils.preprocessing import load_dataset
from utils.recommend import recommend_songs
from utils.youtube_spotify import attach_links
from utils.face_emotion import detect_face_mood
from utils.text_voice import analyze_text_mood
from utils.mood_mapper import sanitize_mood

# --- Flask App Setup ---
app = Flask(__name__)

# Paths
DATA_PATH = "data/final_master_song_dataset.csv"
MODEL_PATH = "model/final_mood_classifier_v1.joblib"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Load Data and Model ---
print("🔄 Loading data and model...")
df = load_dataset(DATA_PATH)
model = joblib.load(MODEL_PATH)
print("✅ Model and dataset loaded successfully.")

# --- Home Route ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        mood_input = None
        language = request.form.get("language", "").strip()
        text_input = request.form.get("user_text", "").strip()
        image = request.files.get("face_image")

        # 🔹 1️⃣ Face detection takes priority
        if image and image.filename != "":
            image_path = os.path.join(UPLOAD_FOLDER, image.filename)
            image.save(image_path)
            mood_input = detect_face_mood(image_path)
            os.remove(image_path)  # cleanup
        # 🔹 2️⃣ Fallback to text-based mood
        elif text_input:
            mood_input = analyze_text_mood(text_input)
        # 🔹 3️⃣ Default mood
        else:
            mood_input = "Calm"

        mood = sanitize_mood(mood_input)
        print(f"🎭 Final detected mood: {mood} | Language: {language or 'Any'}")

        # 🔹 4️⃣ Get recommendations
        recommendations = recommend_songs(df, mood, language)
        recommendations_dict = recommendations.to_dict(orient="records")

        # 🔹 5️⃣ Attach YouTube + Spotify links
        songs_with_links = attach_links(recommendations_dict)

        return render_template(
            "results.html",
            mood=mood,
            language=language or "All",
            songs=songs_with_links,
        )

    return render_template("index.html")


# --- Health Check / API Ping ---
@app.route("/ping")
def ping():
    return {"status": "ok", "message": "Flask backend running successfully 🎶"}

# --- Run App ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
