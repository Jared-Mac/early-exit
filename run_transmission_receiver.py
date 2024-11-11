import socket

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_socket.bind(('localhost', 8080))

# maximum connection
server_socket.listen(5)
print("Server started, waiting...")

while True:

    client_socket, address = server_socket.accept()
    print(f"connection from: {address}")

    data = client_socket.recv(1024).decode()
    print(f"received data: {data}")

    response = "Receiver has got the message"
    client_socket.send(response.encode())

    client_socket.close()
