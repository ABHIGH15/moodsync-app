# utils/face_emotion.py
from fer import FER
import cv2
import numpy as np

EMOTION_TO_MOOD = {
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Energetic",
    "surprise": "Energetic",
    "neutral": "Calm",
    "disgust": "Sad",
    "fear": "Calm"
}

_detector = FER(mtcnn=True)

def detect_emotion_from_file(file_storage) -> str:
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return "Calm"
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    top = _detector.top_emotion(img_rgb)  # (emotion, score)
    if not top:
        return "Calm"
    emotion, _ = top
    return EMOTION_TO_MOOD.get(emotion, "Calm")
