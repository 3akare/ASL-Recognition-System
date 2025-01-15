import cv2
import numpy as np
import matplotlib as plt
import os
import time
import mediapipe as mp

mp_holistics = mp.solutions.holistic # Mediapipe holistics
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


def mediapipe_detection(image, model):
    pass

cap = cv2.VideoCapture(0) #open webcam
while cap.isOpened():
    # Reading from webcam
    ret, frame = cap.read()
    # Show to screen
    cv2.imshow("ASL-Detection", frame)\
    # Exit gracefully
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()