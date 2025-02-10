# Hand Gesture Recognition System

## Overview
This project is a real-time hand gesture recognition system that translates American Sign Language (ASL) gestures into text and audio. It leverages MediaPipe for hand tracking, an LSTM-based deep learning model for gesture classification, and a Flask-based web interface for interaction. The system uses gRPC for efficient communication between the client and the server.

## Features
- **Real-time Gesture Recognition**: Uses a webcam to detect and classify hand gestures.
- **MediaPipe Integration**: Extracts keypoints from hand movements.
- **Deep Learning Model**: An LSTM-based model trained on gesture data.
- **gRPC Communication**: Efficiently transmits gesture data for processing.
- **Flask Web Interface**: Provides a user-friendly UI for interaction.
- **Scalable and Modular**: Well-structured codebase for easy modifications and extensions.

## Folder Structure
```
gesture_recognition/
├── README.md                # Project documentation
├── requirements.txt         # Dependencies
├── signData.proto           # gRPC protocol definition
├── models/                  # Trained models
├── data/                    # Dataset and collected keypoints
├── logs/                    # Training logs
├── common/                  # Shared utilities
│   ├── gesture_utils.py     # MediaPipe-based hand tracking
├── app/                     # Web application
│   ├── routes.py            # Flask API endpoints
│   ├── camera.py            # Webcam handling
│   ├── grpc_client.py       # Communicates with the gRPC server
│   └── templates/           # HTML templates
├── server/                  # gRPC server
│   ├── grpc_server.py       # Server entry point
│   └── service.py           # Model inference logic
└── model/                   # Machine learning model
    ├── model.py             # LSTM architecture
    ├── dataset.py           # Data loading utilities
    ├── train.py             # Model training script
```

## Installation
### Prerequisites
- Python 3.8+
- Virtual environment (recommended)
- TensorFlow, MediaPipe, Flask, and gRPC

### Setup
1. Train the model (optional):
   ```sh
   python model/train.py
   ```
2. Start the gRPC server:
   ```sh
   python server/grpc_server.py
   ```
3. Start the Flask app:
   ```sh
   python app/routes.py
   ```
4. Open the web interface in your browser:
   ```
   http://127.0.0.1:5000/
   ```

## Usage
- Run the application and use your webcam to make hand gestures.
- The system will recognize gestures and display the corresponding text/audio output.
- The web UI provides options for interaction.

