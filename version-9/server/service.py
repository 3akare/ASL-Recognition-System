# server/service.py
import numpy as np
import tensorflow as tf
import os
import signData_pb2
import signData_pb2_grpc

ACTIONS = ["hello", "thanks", "iloveyou"]
label_map = {i: action for i, action in enumerate(ACTIONS)}

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'models', 'action.h5')
model = tf.keras.models.load_model(MODEL_PATH)

class SignDataService(signData_pb2_grpc.StreamDataServiceServicer):
    """
    gRPC service that receives gesture sequences, predicts actions,
    and returns a concatenated string of predictions.
    """
    def biDirectionalStream(self, request, context):
        try:
            sequences = []
            for gesture in request.data:
                # Each gesture.points is a flattened sequence.
                total_length = len(gesture.points)
                sequence_length = 30  # Must match your capture
                feature_size = total_length // sequence_length
                sequence = np.array(gesture.points).reshape((sequence_length, feature_size))
                sequences.append(sequence)
            
            if not sequences:
                return signData_pb2.ResponseMessage(reply="No gesture data received")
            
            sequences = np.array(sequences)
            predictions = model.predict(sequences)
            predicted_actions = [label_map[int(np.argmax(pred))] for pred in predictions]
            response_string = " ".join(predicted_actions)
            return signData_pb2.ResponseMessage(reply=response_string)
        except Exception as e:
            print(f"Error in biDirectionalStream: {e}")
            return signData_pb2.ResponseMessage(reply="Error processing data")

