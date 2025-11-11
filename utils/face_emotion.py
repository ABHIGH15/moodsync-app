# utils/face_emotion.py
import os

def detect_face_mood(image_path: str) -> str:
    """
    Lightweight fallback version for Render (no FER dependency).
    Returns a mock mood label instead of running TensorFlow.
    """
    print("⚠️ FER not available on Render – skipping face analysis.")
    # You can later integrate a cloud API (e.g., DeepFace or Azure Face)
    return "Calm"
