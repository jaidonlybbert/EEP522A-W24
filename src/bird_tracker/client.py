import socket
from array import array
import time
import csv


def connection_init():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn_timeout = time.time() + 20

    while (conn_timeout - time.time() > 0):
        try:
            s.connect(('10.0.0.116', 5007))
            print('Successfully connected')
            return s
        except ConnectionError:
            print('No connection. Attempting reconnection. Reset peer.')
            time.sleep(1)
    print('Connection timed out.')
    return


def main():
    bytes_recvd = s.recv_into(audio_buffer)
    print(audio_buffer)
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(audio_buffer[:(bytes_recvd // 4)])
    return bytes_recvd // 4


if __name__ == '__main__':
    s = connection_init()
    audio_buffer = array('I', [0] * 256)
    filepath = './microphone_reading.csv'
    sample_count = 0
    if s:
        while (sample_count < 100_000):
            sample_count += main()
        print("Wrote 100k samples")
        s.close()
