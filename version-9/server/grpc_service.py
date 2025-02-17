# server/grpc_server.py
from concurrent import futures
import grpc
import time
import signData_pb2_grpc
from service import SignDataService

def serve():
    """
    Start the gRPC server.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    signData_pb2_grpc.add_StreamDataServiceServicer_to_server(SignDataService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("gRPC server started on port 50051")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()

