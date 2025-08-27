import numpy as np
from scipy.io import wavfile

BIT_DEPTH_TO_DTYPE = {
    8: np.uint8,
    16: np.int16,
    32: np.int32,
}

class BasebandFileReader:
    def __init__(self, filename, bits=16, samples_num=5016):
        self._filename = filename
        self._bits = bits
        self._samples_num = samples_num
        self._pos = 0
        self._stop_call = False

        try:
            self._dtype = BIT_DEPTH_TO_DTYPE[bits]
        except KeyError:
            raise ValueError(f"Unsupported bit depth: {bits}")

        self._samplerate, raw_data = wavfile.read(self._filename)

        if np.issubdtype(raw_data.dtype, np.integer):
            scale = float(2**(bits - 1))
            self._data = raw_data.astype(np.float32) / scale
        elif np.issubdtype(raw_data.dtype, np.floating):
            self._data = raw_data.astype(np.float32)
        else:
            raise ValueError("Unsupported WAV data type")

    @property
    def position(self):
        return self._pos

    @property
    def sample_rate(self):
        return self._samplerate

    @property
    def samples_available(self):
        return len(self._data) - self._pos

    @property
    def data(self):
        return self._data.copy()

    def receive_stream(self, callback, chunk_size=1024):
        self._stop_call = False
        chunk_counter = 0

        while self._pos < len(self._data) and not self._stop_call:
            chunk = self._data[self._pos:self._pos + chunk_size]
            callback(chunk.copy())
            self._pos += chunk_size
            chunk_counter += 1

    def receive_samples(self, samplesnum=0):
        num_samples = self._samples_num if samplesnum == 0 else samplesnum
        samples = self._data[self._pos:self._pos + num_samples].copy()
        self._pos += num_samples
        return samples

    def stop(self):
        self._stop_call = True

    def reset(self):
        self._pos = 0
        self._stop_call = False
