from flask import Flask, render_template, request
import pandas as pd

from utils.preprocessing import load_dataset
from utils.mood_mapper import add_mood_column
from utils.face_emotion import detect_emotion_from_file
from utils.text_voice import text_to_mood
from utils.recommend import recommend_songs
from utils.youtube_spotify import attach_links

DATA_PATH = 'data/final_master_song_dataset.csv'

app = Flask(__name__)

# Load dataset
df = load_dataset(DATA_PATH)
df = add_mood_column(df)  # if mood already exists, it just cleans/standardizes

@app.route('/', methods=['GET'])
def home():
    languages = sorted(df['language'].dropna().unique().tolist())
    moods = ['Happy', 'Sad', 'Energetic', 'Calm']
    return render_template('index.html', languages=languages, moods=moods)

@app.route('/recommend', methods=['POST'])
def recommend():
    # Determine mood from face, text, or manual selection
    mood = None

    # 1) Face image
    if 'face' in request.files and request.files['face'].filename:
        mood = detect_emotion_from_file(request.files['face'])

    # 2) Text fallback (if provided)
    if not mood:
        text = request.form.get('feeling_text', '').strip()
        if text:
            mood = text_to_mood(text)

    # 3) Manual dropdown fallback
    if not mood:
        mood = request.form.get('mood', 'Calm')

    language = request.form.get('language', '').strip()

    recs = recommend_songs(df, mood, language, n=8)
    rows = recs.to_dict(orient='records')
    rows = attach_links(rows)

    return render_template('results.html', mood=mood, language=language, songs=rows)

if __name__ == '__main__':
    app.run(debug=True)
