import sys
import socket

class TCPCommunication():
    def __init__(self):
        self.ip = "100.72.193.15"
        self.port = 5000
        self.buff_size = 1024
        self.command_sent = False

    def openServer(self):
        maxConnections = 1

        print("Waiting for connection...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # prevent address already in use error
            try:
                s.bind((self.ip, self.port))
                s.listen(maxConnections)
            
                conn, addr = s.accept()
            except socket.error as msg:
                print("[TCP ERROR]: {}".format(msg))
                sys.exit(1)
            self.s = s

        self.conn = conn
        self.addr = addr
        print("Found connection.")

    def readFromClient(self):
        return self.conn.recv(self.buff_size).decode("utf-8")

    def connectClient(self):
        try:
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.connect((self.ip, self.port))
        except socket.error as msg:
            print("[TCP ERROR]: {}".format(msg))
            sys.exit(2)

    def sendToServer(self, data):
        data = data.encode("utf-8")
        if not self.command_sent:
            self.s.send(data)
            self.command_sent = True

    def closeClientConnection(self):
        self.s.close()