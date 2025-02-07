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
             mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

     # Draw pose landmarks
     mp_drawing.draw_landmarks(
             image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
             mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
             mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))

     # Draw left hand landmarks
     mp_drawing.draw_landmarks(
             image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
             mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
             mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))

     # Draw right hand landmarks
     mp_drawing.draw_landmarks(
             image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
             mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
             mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

def extract_keypoints(results):
    pose = np.array([[res.x ,res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x ,res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    rh =  np.array([[res.x ,res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    lh =  np.array([[res.x ,res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])


cap = cv2.VideoCapture(0) # open webcam
DATA_PATH = os.path.join("MP_Data") # data folder path
actions = np.array(["hello", "thanks", "iloveyou"]) # actions that we try to detect
no_sequences = 30 # 30 videos worth of data
sequence_length = 30 # each video will be 30 frames long


with mp_holistic.Holistic( min_detection_confidence=0.5,  min_tracking_confidence=0.5 ) as holistic:
    # Loop through actions
    for action in actions:
        # Loop through sequences
        for sequence in range(no_sequences):
            # Loop through video length
            for frame_num in range(sequence_length):
                # Read Feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                # Display Flow
                if frame_num == 0:
                    cv2.putText(image, "STARTING COLLECTION", (200, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, "Collecting frames for {} video number {}".format(action, sequence), (20, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow("OpenCV Feed", image)
                    cv2.waitKey(3000)
                else:
                    cv2.putText(image, "Collecting frames for {} video number {}".format(action, sequence), (20, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow("OpenCV Feed", image)

                # Check for exit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exit requested. Cleaning up.")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()  # Immediate exit

                # Extract Keypoints values
                keypoints = extract_keypoints(results)

                # Ensure the directory exists before saving
                dir_path = os.path.join(DATA_PATH, action, str(sequence))
                os.makedirs(dir_path, exist_ok=True)

                # Save the keypoints
                npy_path = os.path.join(dir_path, f"{frame_num}.npy")
                np.save(npy_path, keypoints)

cap.release()
cv2.destroyAllWindows()
