# common/gesture_utils.py
import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Holistic (for face, pose, hands)
mp_holistic = mp.solutions.holistic

def mediapipe_detection(image, model):
    """
    Convert image to RGB, process with MediaPipe holistic, and return the processed image and results.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    """
    Extract keypoints from MediaPipe holistic results and return as a flat numpy array.
    """
    # Pose: 33 landmarks * 4 (x, y, z, visibility)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(33 * 4)
    # Face: 468 landmarks * 3 (x, y, z)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() \
           if results.face_landmarks else np.zeros(468 * 3)
    # Left hand: 21 landmarks * 3 (x, y, z)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
         if results.left_hand_landmarks else np.zeros(21 * 3)
    # Right hand: 21 landmarks * 3 (x, y, z)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
         if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])
