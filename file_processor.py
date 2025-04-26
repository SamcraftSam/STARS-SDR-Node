import numpy as np
from scipy.io import wavfile

class BasebandFileReader:
    def __init__(self, filename, bits=16, samples_num=5016):
        self._filename = filename
        self._bits = bits
        self._samplerate, self._data = wavfile.read(self._filename)
        self._pos = 0
        self._scale = 32767 if self._bits == 16 else (2**(self._bits - 1) - 1)
        self._stop_call = False
        self._samples_num = samples_num

    @property
    def position(self):
        return self._pos

    @property
    def sample_rate(self):
        return self._samplerate

    @property
    def samples_available(self):
        return len(self._data) - self._pos

    def receive_stream(self, callback, chunk_size=1024):
        self._stop_call = False
        if self._data.dtype in (np.float32, np.float64):
            self._data = (self._data * self._scale).astype(np.int16)

        while self._pos < len(self._data) and not self._stop_call:
            chunk = self._data[self._pos:self._pos + chunk_size]
            buf = chunk.astype(np.int16).tobytes()
            callback(buf)
            self._pos += chunk_size

    def receive_samples(self, samplesnum=0):
        if self._data.dtype in (np.float32, np.float64):
            self._data = (self._data * self._scale).astype(np.int16)

        num_samples = self._samples_num if samplesnum == 0 else samplesnum
        return self._data[0:num_samples].copy()  # Return a copy to prevent modification
    
    def stop(self):
        self._stop_call = True

    def reset(self):
        self._pos = 0
        self._stop_call = False