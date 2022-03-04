#!/usr/bin/env python3

from sys import argv
from zlib import compress, decompress
from pathlib import Path
from numpy import arange, argmax, array_split, int16, pi as PI, sin
from numpy.core.multiarray import array, concatenate
from numpy.fft import fft, fftfreq
from scipy.io.wavfile import read, write

SR = 16000
F0 = 450
SYM_RATE = 10
FDEV = 10 # +2550 Hz

PI2 = 2 * PI

FMAX = F0 + FDEV*255

TPS = int(FMAX / SYM_RATE)

# F = 3000
# T = 1/3000 = 0.0003
# tick = 1/16000 = 0.000006
# tps = 450/10 = 45
# tps = 3000/10 = 300

def c2f(c):
    return F0 + c * FDEV


def f2c(f):
    return (f - F0) // FDEV


def encode(msg):
    pi2tsr = PI2 * arange(TPS) / SR
    return concatenate([sin(pi2tsr * c2f(c)) for c in msg])


def decode(signal):
    T = signal.size // TPS
    ff = fftfreq(TPS, 1 / SR)

    freqs = [
        int(abs(ff[argmax(abs(fft(carrier)))]))
        for carrier in array_split(signal, T)
    ]

    return bytes(f2c(f) for f in freqs)


def main(filename):
    src = Path(filename)
    if src.suffix == '.wav':
        encoded = array(read(filename)[1], dtype=float) / 32767
        compressed = decode(encoded)
        decoded = decompress(compressed).decode()
        print(decoded)
    else:
        message = src.read_bytes()
        compressed = compress(message)
        encoded = encode(compressed)
        write('fsk.wav', SR, int16(encoded * 32767))


if __name__ == '__main__':
    main(argv[1])
