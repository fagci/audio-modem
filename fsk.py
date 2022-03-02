#!/usr/bin/env python3

from numpy import arange, argmax, array_split, int16, pi as PI, sin
from numpy.core.multiarray import concatenate
from numpy.fft import fft, fftfreq
from scipy.io.wavfile import write

message = str(arange.__doc__)

SR = 44100
F0 = 440
SYM_RATE = 8
FDEV = 16

PI2 = 2 * PI

TPS = int(SR / SYM_RATE)
MSG_LEN = len(message)

print('Source message:', message)


print('Encoding...')

freqs = [F0 + ord(c) * FDEV for c in message]
print(freqs)

t = arange(TPS)
modulated = concatenate([sin(PI2 * t / SR * f) for f in freqs])

print()

print('Decoding...')

T = modulated.size // TPS
ff = fftfreq(TPS, 1 / SR)

freqs = [
    int(abs(ff[argmax(abs(fft(carrier)))]))
    for carrier in array_split(modulated, T)
]

print(freqs)

decoded = ''.join(chr((f - F0) // FDEV) for f in freqs)

print('Decoded message:', decoded)

write('fsk.wav', SR, int16(modulated * 32767))
