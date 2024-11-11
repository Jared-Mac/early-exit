import socket

def receive_data(sock):
    while True:
        data = sock.recv(1024)  # get 1KB each time
        if not data:
            print("client disconnected")
            break
        print(f"received data: {data}")

# 服务端监听设置
server_address = ('0.0.0.0', 8080)
server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_sock.bind(server_address)
server_sock.listen(1)

print("waiting for client...")
client_sock, client_address = server_sock.accept()
print(f"client connected: {client_address}")

try:
    receive_data(client_sock)
finally:
    client_sock.close()
    server_sock.close()
