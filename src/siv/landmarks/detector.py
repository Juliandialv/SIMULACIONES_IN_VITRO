"""Facial landmark detection using MediaPipe FaceMesh"""

import numpy as np
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh


def detect_landmarks_2d(
        image: np.ndarray,
) -> np.ndarray | None:
    """Detect 468 facial landmarks on a 2D image using MediaPipe FaceMesh.
    
    Args:
        image: BGR image as NumPy array (e. g. loaded with cv2.imread)
        
    Returns:
        Array of shape (468, 2) with (x, y) pixel coordinates,
        or None if no face was detected
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
    ) as face_mesh:
        results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    coords = np.array([[lm.x * w, lm.y * h] for lm in landmarks])  # (468, 2)

    print(f"[detect_landmarks_2d] {len(coords)} landmarks detected.")
    return coords
