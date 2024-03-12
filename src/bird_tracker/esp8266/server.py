from machine import Pin, ADC
import socket
import utime
from array import array

# on-board LED pin
p2 = Pin(2, Pin.OUT)
# ADC pin
adc = ADC(0)
# Buffer to write ADC values to
# Type code 'I' is unsigned int (4 bytes)
# Length is 100
# ADC values are 0-1024
adc_buffer = array('I', [0] * 256)


def connection_init():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('10.0.0.116', 5007))
    print("socket bound")
    s.listen(1)
    conn, addr = s.accept()
    return conn, addr, s


def recv_loop():
    while 1:
        controls_packet = conn.recv(2)
        print(controls_packet)


def main():
    while True:
        populate_buffer()
        # controls_packet = conn.recv(4)    # expect 32 bit packet of control signals
        # print(controls_packet)

        # time_before_send = utime.ticks_us()
        conn.write(adc_buffer)
        # print(utime.ticks_diff(utime.ticks_us(), time_before_send))


conn, addr, s = connection_init()
conn.setblocking(True)

if __name__ == '__main__':
    main()


def populate_buffer():
    p2.on()
    for i, _ in enumerate(adc_buffer):
        adc_buffer[i] = adc.read()
    p2.off()
