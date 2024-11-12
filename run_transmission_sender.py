import socket
import time
import argparse

receiver_address = "172.31.149.34"
receiver_port = 8080

def send_data_with_rate(sock, data, rate_kbps):
    """
    transmit data with specific rate（KB/S)
    :param rate_kbps: rate in KB/s
    """
    rate_bps = rate_kbps * 1024
    chunk_size = 1024
    interval = chunk_size / rate_bps

    try:
        while True:
            sock.sendall(data[:chunk_size])
            time.sleep(interval)
    except KeyboardInterrupt:
        print("transmission interrupted")

def main():
    # Add command line arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--rate', type=int, default=100, help='Specified data transmission rate(kbps)')

    args = parser.parse_args()

    rate = args.rate
    server_address = (receiver_address, receiver_port)
    data_to_send = b'A' * 10240  # 10 KB

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(server_address)

    try:
        send_data_with_rate(sock, data_to_send, rate_kbps=rate)
    finally:
        sock.close()


if __name__ == '__main__':
    main()