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


# Web view of model as it is being trained
log_dir = os.path.join("Logs")
tb_callback = TensorBoard(log_dir=log_dir)

# What is sequential API
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation="relu", input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation="relu"))
model.add(LSTM(64, return_sequences=False, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(actions.shape[0], activation="softmax"))

# multiple class classification model
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
model.fit(x_train, y_train, epochs=100, callbacks=[tb_callback])

res = model.predict(x_test)
if actions[np.argmax(res[4])] == "hello" and actions[np.argmax(y_test[4])] == "hello":
    print("you don't suck")

model.save('action.h5')

# Using this neural network... State of the art models use a number of CNN layers and LSTM,
# Less Data Required
# Faster to train
# Faster Detection


if x_train.shape and x_test.shape and y_test.shape and y_train.shape:
    print("Success!")
