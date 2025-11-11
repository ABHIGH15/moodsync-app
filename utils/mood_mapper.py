# utils/mood_mapper.py
import logging

# --- Unified Mood Space ---
VALID_MOODS = ["Happy", "Sad", "Calm", "Energetic"]

# --- Mapping Tables ---
TEXT_TO_MOOD = {
    "positive": "Happy",
    "neutral": "Calm",
    "negative": "Sad",
}

FACE_TO_MOOD = {
    "angry": "Energetic",
    "disgust": "Sad",
    "fear": "Calm",
    "happy": "Happy",
    "neutral": "Calm",
    "sad": "Sad",
    "surprise": "Energetic",
    "bored": "Sad",
    "tired": "Calm",
    "confused": "Calm",
    "joy": "Happy",
}

# --- Core Mappers ---
def map_text_sentiment_to_mood(sentiment_label: str) -> str:
    """Maps text sentiment (positive/neutral/negative) to unified mood."""
    if not sentiment_label:
        return "Calm"
    mood = TEXT_TO_MOOD.get(sentiment_label.lower(), "Calm")
    logging.debug(f"🗣️ Text sentiment '{sentiment_label}' → mood '{mood}'")
    return mood

def map_face_emotion_to_mood(emotion_label: str) -> str:
    """Maps raw facial emotion to unified mood label."""
    if not emotion_label:
        return "Calm"
    mood = FACE_TO_MOOD.get(emotion_label.lower(), "Calm")
    logging.debug(f"🙂 Face emotion '{emotion_label}' → mood '{mood}'")
    return mood

def sanitize_mood(mood: str) -> str:
    """
    Normalizes any mood label into one of: Happy, Sad, Calm, Energetic.
    Useful for fallback safety and API consistency.
    """
    if not mood or not isinstance(mood, str):
        return "Calm"
    mood = mood.capitalize().strip()
    normalized = mood if mood in VALID_MOODS else "Calm"
    if mood != normalized:
        logging.debug(f"⚙️ Sanitized mood '{mood}' → '{normalized}'")
    return normalized

# --- Local Debug ---
if __name__ == "__main__":
    print(map_face_emotion_to_mood("happy"))       # → Happy
    print(map_text_sentiment_to_mood("positive"))  # → Happy
    print(map_face_emotion_to_mood("bored"))       # → Sad
    print(sanitize_mood("angry"))                  # → Calm
