#!/usr/bin/env python3

from pathlib import Path
from sys import argv
from zlib import compress, decompress

from numpy import arange, argmax, array_split, int16, pi as PI, sin
from numpy.core.multiarray import array, concatenate
from numpy.fft import fft, fftfreq
from scipy.io.wavfile import read, write

SR = 48000
SYM_RATE = 80
F0 = SYM_RATE
FDEV = SYM_RATE

PI2 = 2 * PI

TPS = 1 / SYM_RATE
SPS = SR // SYM_RATE


def c2f(c):
    return F0 + c * FDEV


def f2c(f):
    return (f - F0) // FDEV


def encode(msg):
    pi2tsr = PI2 * arange(0, TPS, 1 / SR)
    return concatenate([sin(pi2tsr * c2f(c)) for c in msg])


def decode(signal):
    N_BYTES = signal.size // SPS

    ff = fftfreq(SPS, 1 / SR)

    freqs = [
        int(abs(ff[argmax(abs(fft(carrier)))]))
        for carrier in array_split(signal, N_BYTES)
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
