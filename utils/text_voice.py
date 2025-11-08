# utils/text_voice.py
from textblob import TextBlob

def text_to_mood(text: str) -> str:
    if not text or not text.strip():
        return "Calm"
    p = TextBlob(text).sentiment.polarity  # -1..1
    if p > 0.3:  return "Happy"
    if p < -0.3: return "Sad"
    return "Calm"
