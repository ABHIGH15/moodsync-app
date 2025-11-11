# utils/text_voice.py
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging
import re

# --- Initialize Sentiment Analyzer ---
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

try:
    analyzer = SentimentIntensityAnalyzer()
except Exception as e:
    logging.warning(f"⚠️ Failed to initialize VADER: {e}")
    analyzer = None

# --- Helper ---
def clean_text(text: str) -> str:
    """Removes emojis, symbols, and extra spaces for better sentiment analysis."""
    if not text:
        return ""
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^A-Za-z\s]", "", text)  # keep alphabets only
    return text.strip().lower()

def analyze_text_mood(user_text: str) -> str:
    """
    Analyzes user's text input and predicts mood using sentiment polarity.
    Returns one of: Happy, Sad, Calm, Energetic.
    Render-safe & lightweight.
    """
    if not user_text or not isinstance(user_text, str) or not analyzer:
        return "Calm"  # fallback

    cleaned = clean_text(user_text)
    if not cleaned:
        return "Calm"

    sentiment = analyzer.polarity_scores(cleaned)
    compound = sentiment.get("compound", 0.0)
    pos = sentiment.get("pos", 0.0)
    neg = sentiment.get("neg", 0.0)

    logging.info(f"🗣️ Text: {user_text} | Compound={compound:.2f}, Pos={pos:.2f}, Neg={neg:.2f}")

    # --- Mood Mapping Logic ---
    if compound >= 0.6:
        return "Happy"
    elif 0.2 <= compound < 0.6 or pos > 0.4:
        return "Energetic"
    elif compound <= -0.4 or neg > 0.5:
        return "Sad"
    else:
        return "Calm"

# --- Local Test Example ---
if __name__ == "__main__":
    samples = [
        "I'm so happy and excited today!",
        "Feeling low and lost...",
        "Everything is peaceful and quiet.",
        "I'm motivated and ready to go!",
        "Life is just... okay right now."
    ]
    for s in samples:
        mood = analyze_text_mood(s)
        print(f"Text: {s} → Mood: {mood}")
