# app/routes.py
import threading
from flask import Blueprint, render_template, jsonify, current_app
from datetime import datetime

# Import our singleton CameraHandler and gRPC client
from camera import camera_handler
from grpc_client import send_gesture_sequences

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Render the main UI."""
    return render_template("index.html")

@main_bp.route('/start', methods=['POST'])
def start_webcam():
    """Start webcam capture if not already running."""
    if not camera_handler.running:
        camera_handler.start()
        return jsonify({"status": "Webcam started"})
    return jsonify({"status": "Webcam already running"})

@main_bp.route('/stop', methods=['POST'])
def stop_webcam():
    """Stop webcam capture."""
    if camera_handler.running:
        camera_handler.stop()
        return jsonify({"status": "Webcam stopped"})
    return jsonify({"status": "Webcam is not running"})

@main_bp.route('/send', methods=['POST'])
def send_data():
    """
    Stop capture, retrieve gesture sequences,
    send them via gRPC, and return the prediction result.
    """
    if camera_handler.running:
        camera_handler.stop()
    
    sequences = camera_handler.get_sequences()
    if not sequences:
        return jsonify({"status": "No gestures to send"})
    
    try:
        response = send_gesture_sequences(sequences, timestamp=datetime.now().isoformat())
        return jsonify({"status": "Data sent successfully", "response": response})
    except Exception as e:
        current_app.logger.error(f"Error sending data: {e}")
        return jsonify({"status": f"Error: {str(e)}"})
