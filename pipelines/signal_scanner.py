if __name__=='__main__':
    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from input.capture import *

from input.capture import *
import numpy as np
import soundfile as sf
from dsptools.dsptools import (WavToComplex, EnvelopeDetector, DecisionMaker)
import logging

logging.basicConfig(level=logging.INFO)

COMMON_SAT_FREQUNCIES={
        137.1e6: "NOAA-18",
        137.9e6: "NOAA-19",
        137.62e6: "NOAA-15"
        }

class SignalDetectorPipeline:
    def __init__(self, central_freq, sat_frequencies=COMMON_SAT_FREQUNCIES, tolerance=10000, snr_threshold=5, bin_size=1000, sample_rate=2000000):
        self.sat_frequencies = sat_frequencies
        self.central_freq = central_freq
        self.tolerance = tolerance
        self.snr_threshold = snr_threshold
        self.bin_size = bin_size
        self.sample_rate = sample_rate

        self.converter = WavToComplex(normalize=True)
        self.envelope_detector = EnvelopeDetector()
       

    def _preprocess(self, iq_data):
        iq_data = self.converter(iq_data)
        iq_data = np.real(iq_data)
        return iq_data

    def _get_bin_frequencies(self, num_bins):
        bin_width = self.sample_rate / num_bins
        bin_frequencies = np.fft.fftfreq(num_bins, d=1/self.sample_rate)
        return bin_frequencies

    def run(self, iq_data):
        iq_data = self._preprocess(iq_data)

        spectrum = np.fft.fft(iq_data)
        amplitude = np.abs(spectrum)
        num_bins = len(amplitude)
        bin_frequencies = self._get_bin_frequencies(num_bins)

        best_snr = -np.inf
        self._best_freq = None
        self._best_sat = None

        for sat_freq, sat_name in self.sat_frequencies.items():
            freq_min = sat_freq - self.tolerance
            freq_max = sat_freq + self.tolerance

            for bin_index, freq in enumerate(bin_frequencies):
                real_freq = self.central_freq + freq
                if freq_min <= real_freq <= freq_max:
                    snr = 10 * np.log10(np.maximum(amplitude[bin_index], 1e-12))
                    if snr > self.snr_threshold and snr > best_snr:
                        best_snr = snr
                        self._best_freq = real_freq
                        self._best_sat = sat_name
                        logging.info(f"Detected new signal from {self._best_sat} at {self._best_freq / 1e6:.3f} MHz with SNR {best_snr:.2f} dB")

        if self._best_freq:
            logging.info(f"Detected signal from {self._best_sat} at {self._best_freq / 1e6:.3f} MHz with SNR {best_snr:.2f} dB")
            return self._best_freq, self._best_sat

        #logging.info("No signal detected.")
        return None
    
    @property
    def best_sat(self):
        return self._best_sat
    
    @property
    def best_freq(self):
        return self._best_freq


if __name__ == "__main__":
    # Настройки
    central_freq = 137.805e6
    block_size = 1024  # Размер блока для псевдореалтайма
    filename = "/home/alex/.config/sdrpp/recordings/baseband_137805000Hz_18-38-26_30-03-2025.wav"

    detector = SignalDetectorPipeline(
        sat_frequencies={
            137.9125e6: "NOAA-18",
            137.1e6: "NOAA-19",
            137.62e6: "NOAA-15"
        },
        central_freq=central_freq,
        sample_rate=2000000
    )

    def callback_fun(data):
        detected = detector.run(data)
        if detected:
            print(f"Signal detected: Frequency {detected[0] / 1e6:.3f} MHz, Satellite: {detected[1]}")
            return False
        else:
            #print("No signal detected.")
            return True

    f = StreamedFileReader(filename)
    f.receive_stream(callback_fun)
    # Стримовая обработка большого файла
    # with sf.SoundFile(filename, 'r') as f:
    #     while True:
    #         block = f.read(block_size, dtype='int16')
    #         if len(block) == 0:
    #             break  # Достигнут конец файла

    #         iq_data = block#block[:, 0] + 1j * block[:, 1]  # Преобразуем стерео в комплексный I/Q
    #         detected = detector.run(iq_data)

    #         if detected:
    #             print(f"Signal detected: Frequency {detected[0] / 1e6:.3f} MHz, Satellite: {detected[1]}")
    #             break
    #         else:
    #             #print("No signal detected.")
    #             pass
