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

T = TPS/SR
TPS_2 = TPS // 2

while ct < len(modulated):
    carrier = modulated[ct:ct + TPS]
    k = arange(TPS)

    f = k/T # 2 sides freq range
    f = f[:TPS_2] # 1 side

    y = fft(carrier)
    y = y[:TPS_2]

    mi = argmax(y)

    f = f[mi]
    print(f)

    decoded.append(chr(int((f - F0) / 4)))

    ct += TPS

print('Decoded:', ''.join(decoded))

write('bpsk.wav', SR, int16(modulated * 32767))
