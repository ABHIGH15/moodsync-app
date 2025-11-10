# utils/face_emotion.py
import cv2
from fer import FER
import numpy as np

# Initialize the FER detector (using MTCNN for better accuracy)
emotion_detector = FER(mtcnn=True)

# Mapping raw facial emotion → system moods
EMOTION_TO_MOOD = {
    "happy": "Happy",
    "angry": "Energetic",
    "fear": "Calm",
    "sad": "Sad",
    "surprise": "Energetic",
    "neutral": "Calm",
    "disgust": "Sad",
}

def detect_face_mood(image_path: str) -> str:
    """
    Detects dominant emotion from a given face image and maps it to system mood.
    Returns the mood label (Happy, Sad, Calm, Energetic).
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Invalid image path or unreadable image.")
        
        # Detect emotions
        results = emotion_detector.detect_emotions(img)
        if not results:
            print("⚠️ No face detected in the image.")
            return "Calm"  # fallback mood

        # Get the emotion dictionary of the first detected face
        emotions = results[0]["emotions"]
        dominant_emotion = max(emotions, key=emotions.get)

        # Map to app mood categories
        mapped_mood = EMOTION_TO_MOOD.get(dominant_emotion, "Calm")
        confidence = emotions[dominant_emotion]

        print(f"Detected facial emotion: {dominant_emotion} ({confidence:.2f}) → Mapped mood: {mapped_mood}")
        return mapped_mood
    except Exception as e:
        print(f"[Face Emotion Error] {e}")
        return "Calm"  # safe fallback

if __name__ == "__main__":
    # Local test (you can replace path with an image on your system)
    test_image = "data/test_face.jpg"
    mood = detect_face_mood(test_image)
    print("Predicted Mood:", mood)

