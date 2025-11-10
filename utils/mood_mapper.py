# utils/mood_mapper.py

# Define unified mood categories for your system
VALID_MOODS = ["Happy", "Sad", "Calm", "Energetic"]

# Fine-grained emotion or sentiment → system-level mood
TEXT_TO_MOOD = {
    "positive": "Happy",
    "neutral": "Calm",
    "negative": "Sad"
}

FACE_TO_MOOD = {
    "angry": "Energetic",
    "disgust": "Sad",
    "fear": "Calm",
    "happy": "Happy",
    "neutral": "Calm",
    "sad": "Sad",
    "surprise": "Energetic"
}

def map_text_sentiment_to_mood(sentiment_label: str) -> str:
    """Maps text sentiment (positive/negative/neutral) to unified mood."""
    return TEXT_TO_MOOD.get(sentiment_label.lower(), "Calm")

def map_face_emotion_to_mood(emotion_label: str) -> str:
    """Maps facial emotion to unified mood."""
    return FACE_TO_MOOD.get(emotion_label.lower(), "Calm")

def sanitize_mood(mood: str) -> str:
    """Ensures mood label matches system-level vocabulary."""
    if not mood:
        return "Calm"
    mood = mood.capitalize()
    return mood if mood in VALID_MOODS else "Calm"

if __name__ == "__main__":
    print(map_face_emotion_to_mood("happy"))  # → Happy
    print(map_text_sentiment_to_mood("positive"))  # → Happy
    print(sanitize_mood("energetic"))  # → Energetic
