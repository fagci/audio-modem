#!/usr/bin/env python3

from zlib import compress, decompress

from numpy import arange, argmax, array_split, int16, pi as PI, sin
from numpy.core.multiarray import concatenate
from numpy.fft import fft, fftfreq
from scipy.io.wavfile import write

message = str(argmax.__doc__)

SR = 44100
F0 = 100
SYM_RATE = 10
FDEV = 10

PI2 = 2 * PI

TPS = int(SR / SYM_RATE)


def c2f(c):
    return F0 + c * FDEV


def f2c(f):
    return (f - F0) // FDEV


def encode(msg):
    t = arange(TPS)
    parts = [sin(PI2 * t / SR * c2f(c)) for c in msg]
    return concatenate(parts)


def decode(signal):
    T = signal.size // TPS
    ff = fftfreq(TPS, 1 / SR)

    freqs = [
        int(abs(ff[argmax(abs(fft(carrier)))]))
        for carrier in array_split(signal, T)
    ]

    return bytes(f2c(f) for f in freqs)


def main():
    print('Source message:', message)

    print('Encoding...')
    compressed = compress(message.encode())
    encoded = encode(compressed)

    print('Decoding...')
    compressed = decode(encoded)
    decoded = decompress(compressed).decode()

    print('Decoded message:', decoded)

    write('fsk.wav', SR, int16(encoded * 32767))

    assert (decoded == message)


if __name__ == '__main__':
    main()
