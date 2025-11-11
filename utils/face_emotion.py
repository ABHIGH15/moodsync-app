# utils/face_emotion.py
import os
import random
import logging

# Supported moods (same as your main app)
MOODS = ["Calm", "Happy", "Sad", "Energetic"]

def detect_face_mood(image_path: str) -> str:
    """
    Lightweight Render-safe fallback.
    - Avoids TensorFlow or FER imports.
    - Returns a pseudo-random mood near 'Calm'.
    - Logs what it’s doing for transparency.
    """
    if not os.path.exists(image_path):
        logging.warning(f"⚠️ Image path does not exist: {image_path}")
        return "Calm"

    # simulate analysis delay for realism (no heavy compute)
    logging.info(f"🖼️ Received face image: {os.path.basename(image_path)}")

    # Since FER is unavailable, choose 'Calm' with mild random variation
    mood = random.choices(
        population=MOODS,
        weights=[0.6, 0.2, 0.1, 0.1],  # mostly calm
        k=1
    )[0]

    logging.info(f"🤖 Face analysis skipped (FER unavailable) → Returning mock mood: {mood}")
    return mood
