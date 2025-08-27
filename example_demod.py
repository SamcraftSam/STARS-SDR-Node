import numpy as np
import sounddevice as sd
import scipy.signal
import wave
import scipy.io.wavfile

# === Настройки ===
input_filename = "/home/alex/.config/sdrpp/recordings/baseband_137805000Hz_20-14-26_30-03-2025.wav"
duration_seconds = 30
target_audio_rate = 48000
center_freq = 137805000  # частота записи
target_freq = 137625000  # целевая частота (например NOAA)
iq_bandwidth = 40000     # ширина сигнала NOAA, 40 кГц

# === Загрузка IQ-данных ===
print("Loading WAV...")
rate, raw = scipy.io.wavfile.read(input_filename)
print(f"Loaded {len(raw)} samples at {rate} Hz")

if raw.ndim != 2 or raw.shape[1] != 2:
    raise ValueError("Input WAV must be stereo with I/Q data.")

iq = raw[:, 0].astype(np.float32) + 1j * raw[:, 1].astype(np.float32)

n_samples = int(rate * duration_seconds)
iq = iq[:n_samples]

# === Смещение частоты ===
freq_shift = -(center_freq - target_freq)
print(f"Frequency shifting by {freq_shift} Hz...")

t = np.arange(len(iq)) / rate
iq_shifted = iq * np.exp(-1j * 2 * np.pi * freq_shift * t)

# === Фильтрация полосы ===
print("Bandpass filtering...")
b, a = scipy.signal.butter(5, [ (iq_bandwidth/2) / (rate/2) ], btype='low')
iq_filtered = scipy.signal.lfilter(b, a, iq_shifted)

# === FM Демодуляция ===
def fm_demodulate(iq):
    phase = np.angle(iq)
    demod = np.diff(np.unwrap(phase))
    return demod

print("Demodulating FM...")
audio = fm_demodulate(iq_filtered)

# === Декадирование ===
decimation_factor = int(round(rate / target_audio_rate))
actual_audio_rate = rate / decimation_factor
print(f"Decimating by factor {decimation_factor} -> {actual_audio_rate} Hz")

audio = scipy.signal.decimate(audio, decimation_factor, ftype='fir')

# === Нормализация ===
audio /= np.max(np.abs(audio))

# === Проигрывание звука ===
print("Playing audio...")
sd.play(audio, int(actual_audio_rate))
sd.wait()

# === Сохранение в WAV ===
output_filename = 'output_audio2.wav'
print("Saving to output WAV...")

audio_int16 = np.int16(audio * 32767)

with wave.open(output_filename, 'w') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(int(actual_audio_rate))
    wf.writeframes(audio_int16.tobytes())

print("Done. Audio saved to", output_filename)
