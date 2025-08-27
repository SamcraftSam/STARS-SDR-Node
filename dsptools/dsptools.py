import numpy as np
import scipy.signal
from scipy.signal import *
from abc import ABC, abstractmethod
from apt_tools.apt_decoder import APT
import sounddevice as sd
import logging
import wave

logging.basicConfig(level=logging.INFO)

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

""" 
    Decoders can inherit library(foreign modules) 
    classes if they were used

    This class more like syntaxis reference than an actual 
    parent for the decoders. 
"""
class GenericDecoder(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, data):
        pass

    @abstractmethod
    def decode(self):
        pass

### DATA CONVERSION ###

class WavToComplex(GenericModule):
    def __init__(self, normalize=False):
        self.normalize = normalize

    def __call__(self, data) -> np.ndarray:
        if data.ndim < 2 or data.shape[1] < 2:
            raise ValueError("WAV data must be stereo with I/Q channels.")
        iq = data[:, 0] + 1j * data[:, 1]

        if self.normalize:
            if np.issubdtype(data.dtype, np.integer):
                max_val = np.iinfo(data.dtype).max
                iq = iq / max_val
            elif not np.issubdtype(data.dtype, np.floating):
                raise TypeError("Unsupported WAV data type for normalization.")

        return iq

class BytesToComplex(GenericModule):
    def __init__(self, data_type=np.int8, normalize=False):
        self.dtype = data_type
        self.normalize = normalize
        self.iq = None

    def __call__(self, data: bytes) -> np.ndarray:
        iq = np.frombuffer(data, dtype=self.dtype)
        
        if len(iq) % 2 != 0:
            raise ValueError("Length of the array must be even")
        
        if self.normalize:
            max_val = np.iinfo(iq.dtype).max
            iq = iq.astype(np.float32) / max_val

        self.iq = iq[::2] + 1j * iq[1::2]
        return self.iq

### FILTERS ###

class LowPassFilterIIR(GenericModule):
    def __init__(self, cutoff_hz, sample_rate_hz, order=5, realtime=True):
        nyq = 0.5 * sample_rate_hz
        norm_cutoff = cutoff_hz / nyq
        self.b, self.a = butter(order, norm_cutoff, btype='low', analog=False)
        self.zi = lfilter_zi(self.b, self.a) * 0
        self.realtime = realtime

    def __call__(self, data):
        if self.realtime:
            y, self.zi = lfilter(self.b, self.a, data, zi=self.zi)
            return y
        else:
            return lfilter(self.b, self.a, data)

class LowPassFilterFIR(GenericModule):
    def __init__(self, cutoff_hz, sample_rate_hz, numtaps=64, realtime=True):
        nyq = 0.5 * sample_rate_hz
        norm_cutoff = cutoff_hz / nyq
        self.taps = firwin(numtaps, norm_cutoff, pass_zero=True)
        self.zi = np.zeros(len(self.taps) - 1)
        self.realtime = realtime

    def __call__(self, data):
        if self.realtime:
            y, self.zi = lfilter(self.taps, 1.0, data, zi=self.zi)
            return y
        else:
            return lfilter(self.taps, 1.0, data)

class BandPassFilterFIR(GenericModule):
    def __init__(self, low_freq, high_freq, sample_rate, numtaps=128, realtime=True):
        self.taps = firwin(numtaps, [low_freq, high_freq], pass_zero=False, fs=sample_rate)
        self.zi = np.zeros(len(self.taps) - 1)
        self.realtime = realtime

    def __call__(self, data):
        if self.realtime:
            y, self.zi = lfilter(self.taps, 1.0, data, zi=self.zi)
            return y
        else:
            return lfilter(self.taps, 1.0, data)

### DEMODULATORS ###

class FMQuadratureDemod(GenericModule):
    def __init__(self, gain=1.0, realtime=True):
        self.gain = gain
        self.prev = 0
        self.realtime = realtime

    def __call__(self, data):
        if self.realtime:
            v = data * np.conj(np.concatenate(([self.prev], data[:-1])))
            self.prev = data[-1]
            return np.angle(v) * self.gain
        else:
            angle = np.angle(data * np.conj(np.roll(data, 1)))
            angle[0] = self.prev
            self.prev = angle[-1]
            return angle * self.gain

### DECIMATORS ###

class Decimator(GenericModule):
    def __init__(self, factor, ftype='fir'):
        if factor < 1:
            raise ValueError("Decimation factor must be >= 1")
        self.factor = factor
        self.ftype = ftype

    def __call__(self, data):
        return decimate(data, self.factor, ftype=self.ftype)

class AutoDecimator(Decimator):
    def __init__(self, input_rate, target_rate, ftype='fir'):
        if input_rate <= 0 or target_rate <= 0:
            raise ValueError("Sample rates must be positive.")
        if target_rate > input_rate:
            raise ValueError("Target rate must be less than input rate.")

        self.factor = int(round(input_rate / target_rate))
        self.actual_rate = input_rate / self.factor

        if not np.isclose(self.actual_rate, target_rate, rtol=0.05):
            logging.warning(f"Actual decimated rate {self.actual_rate} Hz differs from target {target_rate} Hz.")

        super().__init__(self.factor, ftype)

    def __call__(self, data):
        return super().__call__(data)

### RESAMPLERS ###

class Resampler(GenericModule):
    def __init__(self, up, down):
        self._up = up
        self._down = down

    def __call__(self, data):
        return resample_poly(data, self._up, self._down)

class AutoResampler(Resampler):
    def __init__(self, input_rate, target_rate):
        if input_rate <= 0 or target_rate <= 0:
            raise ValueError("Sample rates must be positive.")

        self.input_rate = input_rate
        self.target_rate = target_rate
        self.gcd = np.gcd(int(input_rate), int(target_rate))
        self.up = int(target_rate // self.gcd)
        self.down = int(input_rate // self.gcd)

        super().__init__(self.up, self.down)
        
### MISC ###

class FrequencyShift(GenericModule):
    def __init__(self, freq_shift_hz, sample_rate):
        self.freq_shift_hz = freq_shift_hz
        self.sample_rate = sample_rate
        self.sample_index = 0

    def __call__(self, data):
        t = np.arange(len(data)) / self.sample_rate
        shift = np.exp(-2j * np.pi * self.freq_shift_hz * (self.sample_index + t))
        self.sample_index += len(data)
        return data * shift

class dBmSignalPower(GenericModule):
    def __init__(self, reference_pwr=-90.0):
        self.ref_p = reference_pwr

    def __call__(self, data):
        strength = np.sum(np.abs(data)**2) / len(data)
        dBm = 10 * np.log10(strength) + self.ref_p
        logging.info(f"Signal Pwr: {dBm}")
        return dBm

### AUDIO ###

class AudioNormalize(GenericModule):
    def __init__(self):
        pass

    def __call__(self, data):
        data /= np.max(np.abs(data))
        return data

### DECODERS ###

class DecoderAPT(APT):
    RATE = 20800
    NOAA_LINE_LENGTH = 2080

    def __init__(self, in_file=None, samplerate=RATE):
        self._in = in_file
        self._signal = []

        if samplerate != self.RATE:
            raise Exception("Resample audio to {}".format(self.RATE))

        if self._in is not None:
            super().__init__(self._in)
         
    def __call__(self, data):
        self._signal.append(data)
        
    def decode(self, outfile=None, imgshow=False):
        if self._in is None:
            all_signal = np.concatenate(self._signal)
            truncate = self.RATE * int(len(all_signal)//self.RATE)
            if truncate > 0:
                all_signal = all_signal[:truncate]
            return super().decode(signal=all_signal, outfile=outfile, imgshow=imgshow)
        return super().decode(outfile=outfile, imgshow=imgshow)


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

class FileAudioSink(GenericSink):
    def __init__(self, filename, sample_rate=48000):
        self.filename = filename
        self.sample_rate = sample_rate
        self.buffer = []

    def __call__(self, data):
        self.buffer.append(data)

    def flush(self):
        full = np.concatenate(self.buffer)
        full = (full / np.max(np.abs(full)) * 32767).astype(np.int16)
        with wave.open(self.filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(full.tobytes())

class StreamAudioSink(GenericSink):
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        self.stream = sd.OutputStream(samplerate=sample_rate, channels=1)
        self.stream.start()

    def __call__(self, data):
        self.stream.write(data.astype(np.float32))

    def flush(self):
        self.stream.stop()
        self.stream.close()

class BufferAudioSink(GenericSink):
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        self.buffer = []

    def __call__(self, data):
        self.buffer.append(data)

    def flush(self):
        all_data = np.concatenate(self.buffer)
        sd.play(all_data, self.sample_rate)
        sd.wait()
