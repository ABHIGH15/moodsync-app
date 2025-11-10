# utils/face_emotion.py
import cv2
from fer import FER
import numpy as np

def detect_face_mood(image_path: str) -> str:
    """
    Detects mood from a face image using FER library.
    Returns one of: ['Happy', 'Sad', 'Angry', 'Surprise', 'Neutral', 'Disgust', 'Fear']
    """
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("❌ Image could not be read. Invalid path or format.")
        
        detector = FER(mtcnn=True)
        emotions = detector.detect_emotions(img)

        if not emotions:
            return "Neutral"

        # Extract top emotion by confidence
        top_emotion, score = detector.top_emotion(img)
        if score < 0.5:
            return "Neutral"

        # Normalize to standard moods
        mood_map = {
            "happy": "Happy",
            "sad": "Sad",
            "angry": "Energetic",
            "neutral": "Calm",
            "surprise": "Happy",
            "disgust": "Sad",
            "fear": "Sad"
        }
        return mood_map.get(top_emotion.lower(), "Calm")

    except Exception as e:
        print("⚠️ Face detection error:", e)
        return "Calm"
