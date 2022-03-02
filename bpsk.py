#!/usr/bin/env python3

from numpy import arange, argmax, int16, linspace, pi as PI, sin
from numpy.core.multiarray import concatenate
from numpy.fft import fft
from scipy.io.wavfile import write

SR = 16000
F0 = 1000

SYM_RATE = 100

PI2 = 2 * PI

TPS = int(SR / SYM_RATE)

message = 'a f k p u'

print('Source message:', message)

MSG_LEN = len(message)

t = arange(0, TPS)

print(f'{TPS=}')

freqs = [F0 + ord(c) * 4 for c in message]
print(freqs)

# encode message (thing its ok by analyzer)
modulated = concatenate([sin(PI2 * t / SR * f) for f in freqs])

decoded = []

ct = 0

print('Decoding...')

# decode
while ct < len(modulated):
    carrier = modulated[ct:ct + TPS]

    xf = linspace(0, SR / 2, TPS // 2)
    mi = argmax(2 / TPS * abs(fft(carrier)[0:TPS // 2]))

    f = xf[mi]
    print(f)

    decoded.append(chr(int((f - F0) / 4)))

    ct += TPS

print('Decoded:', ''.join(decoded))

write('bpsk.wav', SR, int16(modulated * 32767))
