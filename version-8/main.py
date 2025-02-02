import cv2
import numpy as np
import matplotlib as plt
import os
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic # Mediapipe holistic
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Converts the image from BGR (OpenCV's default color format) to RGB (required by MediaPipe)
    image.flags.writeable = False
    results = model.process(image) # Make prediction
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Converts the image from RGB to BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose landmarks
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand landmarks
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand landmarks

def draw_styled_landmarks(image, results):
    # Draw face landmarks
     mp_drawing.draw_landmarks(
             image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
             mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
             mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
             )

     # Draw pose landmarks
     mp_drawing.draw_landmarks(
             image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
             mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
             mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
             )

     # Draw left hand landmarks
     mp_drawing.draw_landmarks(
             image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
             mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
             mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
)

     # Draw right hand landmarks
     mp_drawing.draw_landmarks(
             image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
             mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
             mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
)

#stopped at 40:42

cap = cv2.VideoCapture(0) # open webcam

with mp_holistic.Holistic( min_detection_confidence=0.5,  min_tracking_confidence=0.5 ) as holistic:
    while cap.isOpened():
        # Reading from webcam
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # Show to screen
        cv2.imshow("ASL Detection", image)

        # Exit gracefully
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
