# app/camera.py
import cv2
import threading
import mediapipe as mp
from common import gesture_utils

class CameraHandler:
    """
    Manages webcam capture and keypoint extraction.
    Accumulates frames into fixed-length sequences.
    """
    def __init__(self, sequence_length=30):
        self.running = False
        self.thread = None
        self.buffer = []      # Buffer for current sequence
        self.sequences = []   # Collected complete sequences
        self.lock = threading.Lock()
        self.sequence_length = sequence_length
        
        # Initialize MediaPipe Holistic model
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def start(self):
        """Start the capture thread."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._capture, daemon=True)
            self.thread.start()
    
    def stop(self):
        """Stop the capture thread and release resources."""
        self.running = False
        if self.thread is not None:
            self.thread.join()
        self.holistic.close()
    
    def _capture(self):
        """Capture frames, process keypoints, and form sequences."""
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Process frame with MediaPipe
            _, results = gesture_utils.mediapipe_detection(frame, self.holistic)
            keypoints = gesture_utils.extract_keypoints(results)
            
            with self.lock:
                self.buffer.append(keypoints)
                if len(self.buffer) == self.sequence_length:
                    self.sequences.append(list(self.buffer))
                    self.buffer = []
            
            # Optional: show webcam feed for debugging (remove in production)
            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
        cap.release()
        cv2.destroyAllWindows()
    
    def get_sequences(self):
        """Return and clear collected sequences."""
        with self.lock:
            seqs = self.sequences.copy()
            self.sequences = []
        return seqs

# Create a singleton instance for use across the app
camera_handler = CameraHandler(sequence_length=30)

