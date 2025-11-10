# utils/text_voice.py
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download required lexicon (only once)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except:
    nltk.download('vader_lexicon')

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def analyze_text_mood(user_text: str) -> str:
    """
    Analyzes user's text input and predicts mood based on sentiment.
    Returns one of: Happy, Sad, Calm, Energetic
    """
    if not user_text or not isinstance(user_text, str):
        return "Calm"  # safe fallback

    sentiment = analyzer.polarity_scores(user_text)
    compound = sentiment["compound"]

    if compound >= 0.5:
        return "Happy"
    elif compound >= 0.1:
        return "Energetic"
    elif compound <= -0.4:
        return "Sad"
    else:
        return "Calm"

if __name__ == "__main__":
    # Test locally
    samples = [
        "I'm so happy and excited today!",
        "Feeling low and lost...",
        "Everything is peaceful and quiet.",
        "I'm motivated and ready to go!"
    ]
    for s in samples:
        mood = analyze_text_mood(s)
        print(f"Text: {s} → Mood: {mood}")

