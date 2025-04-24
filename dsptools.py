import numpy as np
from scipy.signal import *
from abc import ABC, abstractmethod
import sounddevice as sd

### ABSTRACT ###

class GenericModule(ABC):
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def __call__(self, data):
        pass

class GenericSink(ABC):
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def __call__(self, data):
        pass
    @abstractmethod
    def flush(self):
        pass

### DATA CONVERSION ###

class WavToComplex(GenericModule):
    def __init__(self, normalize=False):
        self.normalize = normalize
        pass

    def __call__(self, data) -> np.ndarray:
        self.iq = data[:, 0] + 1j * data[:, 1]

        if self.normalize:
            if np.issubdtype(data.dtype, np.integer):
                max_val = np.iinfo(data.dtype).max
                iq = iq / max_val
            elif np.issubdtype(data.dtype, np.floating):
                pass
            else:
                raise TypeError("Unsupported WAV data type for normalization.")

        return self.iq
    
class BytesToComplex(GenericModule):
    def __init__(self, data_type=np.int8, normalize=False):
        self.dtype = data_type
        self.normalize = normalize

    def __call__(self, data: bytes) -> np.ndarray:
        iq = np.frombuffer(data, dtype=self.dtype)
        
        if len(iq) % 2 != 0:
            raise ValueError("Length of the array must be even")
        
        if self.normalize:
            max_val = np.iinfo(iq.dtype).max
            iq = iq.astype(np.float32)/ max_val


        self.iq = iq[::2] + 1j * iq[1::2]

        return self.iq

### FILTERS ###

class LowPassFilter(GenericModule):
    def __init__(self, cutoff_hz, sample_rate_hz, order=5):
        nyq = 0.5 * sample_rate_hz
        norm_cutoff = cutoff_hz / nyq
        self.b, self.a = butter(order, norm_cutoff, btype='low', analog=False)

    def __call__(self, data):
        return lfilter(self.b, self.a, data)

### DEMODULATORS ###

class FMQuadratureDemod(GenericModule):
    def __init__(self, gain=1.0):
        self.gain = gain
        self.prev = 0

    def __call__(self, data):
        angle = np.angle(data * np.conj(np.roll(data, 1)))
        angle[0] = self.prev
        self.prev = angle[-1]
        return angle * self.gain

### MISC ###

class dBmSignalPower(GenericModule):
    def __init__(self, reference_pwr=-90.0):
        self.ref_p = reference_pwr

    def __call__(self, data):
        strength = np.sum(np.abs(data)**2)/len(data)
        dBm = 10*np.log10(strength) + self.ref_p

        return dBm

### OUTPUT (SINKS) ###


class FileSink(GenericSink):
    def __init__(self, path):
        self.path = path
        self.buffer = []

    def __call__(self, data):
        self.buffer.append(data)

    def flush(self):
        with open(self.path, 'wb') as f:
            for block in self.buffer:
                f.write(block.tobytes())

class AudioSink(GenericSink):
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def __call__(self, data):
        sd.play(data, self.sample_rate)


