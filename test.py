import numpy as np
import matplotlib.pyplot as plt

# Параметры
Fs = 1000  # Частота дискретизации
T = 1.0 / Fs
t = np.arange(0, 1, T)  # Время от 0 до 1 секунды
f0 = 50  # Частота сигнала

# Синусоидальный сигнал с частотой 50 Гц
x = 0.5 * np.sin(2 * np.pi * f0 * t)

# FFT
X = np.fft.fft(x)
frequencies = np.fft.fftfreq(len(x), T)

# Амплитудный спектр
amplitude = np.abs(X)

# Только положительные частоты
positive_freqs = frequencies[:len(frequencies)//2]
positive_amplitude = amplitude[:len(amplitude)//2]

# Построение графика
plt.plot(positive_freqs, positive_amplitude)
plt.title('Амплитудный спектр сигнала')
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда')
plt.show()
