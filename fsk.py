#!/usr/bin/env python3

from numpy import arange, argmax, int16, pi as PI, sin, array_split
from numpy.core.multiarray import concatenate
from numpy.fft import fft, fftfreq
from scipy.io.wavfile import write

SR = 44100
F0 = 440

SYM_RATE = 8 # TODO: make faster version work

PI2 = 2 * PI

FMUL = 16

TPS = int(SR / SYM_RATE)

message = '0Z'
message = 'NumPy is the fundamental package for scientific computing in Python'

print('Source message:', message)

MSG_LEN = len(message)

t = arange(0, TPS)

print('Encoding...')

freqs = [F0 + ord(c) * FMUL for c in message]
print(freqs)

modulated = concatenate([
    sin(PI2 * t / SR * f)
    for f in freqs
])

print('Decoding...')

T = modulated.size // TPS
ff = fftfreq(TPS, 1/SR)

freqs = [
    int(abs(ff[argmax(abs(fft(carrier)))]))
    for carrier in array_split(modulated, T)
]

print(freqs)

decoded = ''.join(chr(int((f - F0) / FMUL)) for f in freqs)

print('Decoded message:', decoded)

write('fsk.wav', SR, int16(modulated * 32767))
