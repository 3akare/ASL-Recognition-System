import os
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# Stopped 1:35:00

actions = np.array(["hello", "thanks", "iloveyou"])  # Actions to detect
label_map = {label: num for num, label in enumerate(actions)}  # Label mapping

no_sequences = 30  # Number of video samples per action
sequence_length = 30  # Frames per video
DATA_PATH = os.path.join("MP_Data")  # Data folder path

sequences, labels = [], []

print("Started Processing Data")
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

x = np.array(sequences)
y = to_categorical(labels).astype(int)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)

if x_train.shape and x_test.shape and y_test.shape and y_train.shape:
    print("Success!")
