# app/grpc_client.py
import grpc
import signData_pb2
import signData_pb2_grpc

def send_gesture_sequences(sequences, timestamp):
    """
    Sends collected gesture sequences to the gRPC server for prediction.
    
    Args:
        sequences (list): List of gesture sequences (each is a list of frames).
        timestamp (str): Timestamp string.
    
    Returns:
        str: Prediction reply from the gRPC server.
    """
    # Connect to the gRPC server (adjust host/port as needed)
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = signData_pb2_grpc.StreamDataServiceStub(channel)
        gestures = []
        # For each collected sequence, flatten the data to one list.
        # (The server will reshape to (30, feature_size).)
        for sequence in sequences:
            flat_sequence = [value for frame in sequence for value in frame]
            gesture = signData_pb2.Gesture(points=flat_sequence)
            gestures.append(gesture)
        
        request = signData_pb2.RequestMessage(data=gestures, timestamp=timestamp)
        response = stub.biDirectionalStream(request)
        return response.reply

