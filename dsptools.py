import numpy as np
from scipy.signal import *
from abc import ABC, abstractmethod

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

class LowPassFilter(GenericModule):
    def __init__(self, cutoff_hz, sample_rate_hz, order=5):
        nyq = 0.5 * sample_rate_hz
        norm_cutoff = cutoff_hz / nyq
        self.b, self.a = butter(order, norm_cutoff, btype='low', analog=False)

    def __call__(self, data):
        return lfilter(self.b, self.a, data)


class FMQuadratureDemod(GenericModule):
    def __init__(self, gain=1.0):
        self.gain = gain
        self.prev = 0

    def __call__(self, data):
        angle = np.angle(data * np.conj(np.roll(data, 1)))
        angle[0] = self.prev
        self.prev = angle[-1]
        return angle * self.gain


class PacketDecoder(GenericModule):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, data):
        return (data > self.threshold).astype(np.uint8)


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
