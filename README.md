# moodsync-app

Built by Team Predix

MoodSync is an intelligent music recommendation system that suggests songs based on your mood and preferred language.
It uses lightweight NLP for text-based emotion detection and a structured dataset for mood-driven recommendations — all optimized for Render deployment (no TensorFlow required).

🚀 Features

✅ Mood Detection Options

Upload a face image (mock detection on Render — returns Calm)

Type how you feel — sentiment analysis via NLTK

Choose mood manually (fallback)

✅ Dynamic Recommendations

Suggests songs by mood + language

“Top 5 Mix” + “More you might like”

Live YouTube + Spotify buttons

Shuffle button to reshuffle songs dynamically

✅ Render-Optimized Deployment

Lightweight build (no TensorFlow)

/ping endpoint for health checks

Python 3.10 fixed with render.yaml

🧠 Supported Categories
Moods	Languages
Calm	English
Happy	Hindi
Sad	Tamil
Energetic	Marathi
	Punjabi
🗂️ Project Structure
moodsync-app/
│
├── app.py                       # Flask backend
├── requirements.txt              # All dependencies
├── render.yaml                   # Render deploy configuration
│
├── data/
│   └── final_master_song_dataset.csv
│
├── model/
│   └── final_mood_classifier_v1.joblib
│
├── templates/
│   ├── index.html                # Homepage form
│   └── results.html              # Recommendations page
│
└── utils/
    ├── preprocessing.py
    ├── recommend.py
    ├── youtube_spotify.py
    ├── face_emotion.py
    ├── text_voice.py
    └── mood_mapper.py

⚙️ Local Setup (For Testing)

Clone the repository

git clone https://github.com/ABHIGH2025/moodsync-app.git
cd moodsync-app


Create and activate virtual environment

python -m venv venv
venv\Scripts\activate  # on Windows
# or
source venv/bin/activate  # on macOS/Linux


Install dependencies

pip install -r requirements.txt


Run Flask app locally

python app.py


Visit → http://127.0.0.1:5000

🌐 Deploying to Render

Push your project to GitHub

Create a New Web Service on Render

Connect your GitHub repo

Render automatically detects the render.yaml file

Deploy 🚀

Your app will auto-build and deploy using:

Python 3.10.14
pip install -r requirements.txt
gunicorn app:app


Health check path: /ping

🩵 Example Output

/ping (Render health status endpoint):

{
  "status": "ok",
  "message": "MoodSync backend running successfully 🎶",
  "moods": ["Calm", "Happy", "Sad", "Energetic"],
  "languages": ["English", "Hindi", "Tamil", "Marathi", "Punjabi"],
  "dataset_rows": 1200,
  "team": "Predix"
}

👨‍💻 Authors — Team Predix

Amit Gupta

[Add other team members here]

🧩 Tech Stack

Backend: Flask (Python 3.10)

Frontend: HTML5 + TailwindCSS

ML/NLP: NLTK, TextBlob, Scikit-Learn

Deployment: Render (Free Tier, Gunicorn Server)

🛡️ License

This project is for educational and research purposes under Team Predix.
All song data references are used for demonstration only.
