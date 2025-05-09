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

    """
    GenericModule class: Module for signal processing within DSP pipelines.

    This class is used as a building block within an SDR pipeline to perform specific 
    transformations or processing on the input signal.

    Notes
    -----
    Typically, instances of this class are chained together in a pipeline to perform 
    sequential signal transformations.
    """
    @abstractmethod
    def __init__(self):

        """
        __init__ method: Core operation of the GenericModule class.

        Notes
        -----
        Parameters/configuration goes here
        """
        pass

    @abstractmethod
    def __call__(self, data):

        """
        __call__ method: Core operation of the GenericModule class.

        Notes
        -----
        Real-time processing goes here
        """
        pass

class GenericSink(ABC):

    """
    GenericSink class: Module for signal processing within DSP pipelines.

    This class is used as a building block within an SDR pipeline to perform specific 
    transformations or processing on the input signal.

    Notes
    -----
    Last stage of the pipeline. Save/replay/plot the data
    """
    @abstractmethod
    def __init__(self):

        """
        __init__ method: Core operation of the GenericSink class.

        Notes
        -----
        Configuration goes here
        """
        pass

    @abstractmethod
    def __call__(self, data):

        """
        __call__ method: Core operation of the GenericSink class.

        Notes
        -----
        Real-time processing or data buffering goes here.
        """
        pass

    @abstractmethod
    def flush(self):

        """
        flush method: Core operation of the GenericSink class.


        Notes
        -----
        Offline processing in case real time is not possible.
        """
        pass

class GenericDecoder(ABC):

    """
    GenericDecoder class: Module for signal processing within DSP pipelines.

    This class is used as a building block within an SDR pipeline to perform specific 
    transformations or processing on the input signal.

    Notes
    -----
    Decoders can inherit class from the extrenal library(module) 
    This class more like syntaxis reference than an actual 
    parent for the decoders. 
    """
    @abstractmethod
    def __init__(self):

        """
        __init__ method: Core operation of the GenericDecoder class.

        Notes
        -----
        Set-up the parameters/config
        """
        pass

    @abstractmethod
    def __call__(self, data):

        """
        __call__ method: Core operation of the GenericDecoder class.

        Notes
        -----
        Decode/buffer the data in real time.
        """
        pass

    @abstractmethod
    def decode(self):

        """
        decode method: Core operation of the GenericDecoder class.

        Notes
        -----
        Decode data offline, if real time processing is not possible.
        The processing method varies based on the module's function, whether it's 
        data conversion, signal demodulation, or noise suppression.
        """
        pass

### DATA CONVERSION ###

class WavToComplex(GenericModule):

    """
    WavToComplex class: Converts stereo WAV data to complex I/Q format.

    This class takes a stereo WAV file (I/Q data) and converts it into a complex 
    format suitable for further DSP processing. Typically used in satellite signal 
    processing where data is stored as baseband I/Q pairs.

    Parameters
    ----------
    normalize : Bool
        normalizes signal between [-1;1] if True
    
    Notes
    -----
    This module is commonly used in SDR pipelines to convert recorded baseband data 
    into a format compatible with complex DSP operations.
    """
    def __init__(self, normalize=False):

        """
        __init__ method: Core operation of the WavToComplex class.


        """
        self.normalize = normalize

    def __call__(self, data) -> np.ndarray:

        """
        __call__ method: Core operation of the WavToComplex class.

        """
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

    """
    BytesToComplex class: Converts raw byte data to complex I/Q format.

    This class takes raw bytes from an SDR stream and converts them to complex values. 
    Often used in real-time signal processing when data arrives as byte streams.

    Parameters
    ----------
    data_type : numpy data type, optional
        Specifies the data type of the input bytes (e.g., np.int8).
    normalize : bool, optional
        If True, normalizes the data to the range [-1, 1].

    Returns
    -------
    ndarray
        Complex I/Q signal obtained from byte data.

    Notes
    -----
    Useful when processing raw SDR data that is streamed or stored in byte format.
    """
    def __init__(self, data_type=np.int8, normalize=False):

        """
        __init__ method: Core operation of the BytesToComplex class.

        """
        self.dtype = data_type
        self.normalize = normalize
        self.iq = None

    def __call__(self, data: bytes) -> np.ndarray:

        """
        __call__ method: Core operation of the BytesToComplex class.

        """
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

    """
    LowPassFilterIIR class: Module for signal processing within DSP pipelines.

    This class is used as a building block within an SDR pipeline to perform specific 
    transformations or processing on the input signal.

    Parameters
    ----------
        cutoff_hz : int
            band with freq > cutoff_hz will be filtered out
        
        sample_rate_hz : int
            Sample rate of the signal

        order : int
            Read the scipy butter() docstrings
        
        realtime : Bool
            set True if data is streamed, to save filter coeficient between chunks

    Returns
    -------
    ndarray

    Notes
    -----
    Faster than FIR but not as sharp
    """
    def __init__(self, cutoff_hz, sample_rate_hz, order=5, realtime=True):

        """
        __init__ method: Core operation of the LowPassFilterIIR class.

        """
        nyq = 0.5 * sample_rate_hz
        norm_cutoff = cutoff_hz / nyq
        self.b, self.a = butter(order, norm_cutoff, btype='low', analog=False)
        self.zi = lfilter_zi(self.b, self.a) * 0
        self.realtime = realtime

    def __call__(self, data):

        """
        __call__ method: Core operation of the LowPassFilterIIR class.

        """
        if self.realtime:
            y, self.zi = lfilter(self.b, self.a, data, zi=self.zi)
            return y
        else:
            return lfilter(self.b, self.a, data)

class LowPassFilterFIR(GenericModule):

    """
    LowPassFilterFIR class: Module for signal processing within DSP pipelines.

    This class is used as a building block within an SDR pipeline to perform specific 
    transformations or processing on the input signal.

    Parameters
    ----------
        cutoff_hz : int
            band with freq > cutoff_hz will be filtered out
        
        sample_rate_hz : int
            Sample rate of the signal

        numtaps : int
            Read the scipy firwin() docstrings
        
        realtime : Bool
            set True if data is streamed, to save filter coeficient between chunks

    Returns
    -------
    ndarray

    Notes
    -----
    Filters better than butter, but slower
    """
    def __init__(self, cutoff_hz, sample_rate_hz, numtaps=64, realtime=True):

        """
        __init__ method: Core operation of the LowPassFilterFIR class.

        """
        nyq = 0.5 * sample_rate_hz
        norm_cutoff = cutoff_hz / nyq
        self.taps = firwin(numtaps, norm_cutoff, pass_zero=True)
        self.zi = np.zeros(len(self.taps) - 1)
        self.realtime = realtime

    def __call__(self, data):

        """
        __call__ method: Core operation of the LowPassFilterFIR class.

        """
        if self.realtime:
            y, self.zi = lfilter(self.taps, 1.0, data, zi=self.zi)
            return y
        else:
            return lfilter(self.taps, 1.0, data)

class BandPassFilterFIR(GenericModule):

    """
    BandPassFilterFIR class: Module for signal processing within DSP pipelines.

    This class is used as a building block within an SDR pipeline to perform specific 
    transformations or processing on the input signal.

   Parameters
    ----------
        low_freq : int
            frequencies below will be filtered out

        high_freq : int
            frequencies above will be filtered out
        
        sample_rate_hz : int
            Sample rate of the signal

        numtaps : int
            Read the scipy firwin() docstrings
        
        realtime : Bool
            set True if data is streamed, to save filter coeficient between chunks

    Returns
    -------
    ndarray

    Notes
    -----
    Frequencies below low_freq and above gigh_freq will be filtered out
    """
    def __init__(self, low_freq, high_freq, sample_rate, numtaps=128, realtime=True):

        """
        __init__ method: Core operation of the BandPassFilterFIR class.

        """
        self.taps = firwin(numtaps, [low_freq, high_freq], pass_zero=False, fs=sample_rate)
        self.zi = np.zeros(len(self.taps) - 1)
        self.realtime = realtime

    def __call__(self, data):

        """
        __call__ method: Core operation of the BandPassFilterFIR class.

        """
        if self.realtime:
            y, self.zi = lfilter(self.taps, 1.0, data, zi=self.zi)
            return y
        else:
            return lfilter(self.taps, 1.0, data)

### DEMODULATORS ###

class FMQuadratureDemod(GenericModule):

    """
    FMQuadratureDemod class: Module for signal processing within DSP pipelines.

    This class is used as a building block within an SDR pipeline to perform specific 
    transformations or processing on the input signal.

    Parameters
    ----------
        gain : float
            gain of the signal
        
        realtime : Bool
            set True if processing in realtime

    Returns
    -------
    ndarray

    Notes
    -----
    Applies Frequency Demodulation to the signal.
    """
    def __init__(self, gain=1.0, realtime=True):

        """
        __init__ method: Core operation of the FMQuadratureDemod class.

        """
        self.gain = gain
        self.prev = 0
        self.realtime = realtime

    def __call__(self, data):

        """
        __call__ method: Core operation of the FMQuadratureDemod class.

        """
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

    """
    Decimator class: Module for signal processing within DSP pipelines.

    This class is used as a building block within an SDR pipeline to perform specific 
    transformations or processing on the input signal.

    Parameters
    ----------
        factor : int
            read scipy decimate() docs

        ftype : String
            read scipy decimate() docs
        data : ndarray

    Returns
    -------
    ndarray

    Notes
    -----
    Typically, instances of this class are chained together in a pipeline to perform 
    sequential signal transformations.
    """
    def __init__(self, factor, ftype='fir'):

        """
        __init__ method: Core operation of the Decimator class.

        """
        if factor < 1:
            raise ValueError("Decimation factor must be >= 1")
        self.factor = factor
        self.ftype = ftype

    def __call__(self, data):

        """
        __call__ method: Core operation of the Decimator class.

        """
        return decimate(data, self.factor, ftype=self.ftype)

class AutoDecimator(Decimator):

    """
    Decimates the signal. User-friendly version

    Parameters
    ----------
        input_rate : int
        target_rate : int
        ftype: String
            Read the docs
        data : ndarray

    Returns
    -------
    ndarray

    Notes
    -----
    Typically, instances of this class are chained together in a pipeline to perform 
    sequential signal transformations.
    """
    def __init__(self, input_rate, target_rate, ftype='fir'):

        """
        __init__ method: Core operation of the AutoDecimator class.

        """
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

        """
        __call__ method: Core operation of the AutoDecimator class.

        """
        return super().__call__(data)

### RESAMPLERS ###

class Resampler(GenericModule):

    """
    Changes samplerate of the signal

    Parameters
    ----------
        up : int
        down : int
        data : ndarray

    Returns
    -------
    ndarray

    Notes
    -----
    Use AutoResampler instead
    """
    def __init__(self, up, down):

        """
        __init__ method: Core operation of the Resampler class.

        """
        self._up = up
        self._down = down

    def __call__(self, data):

        """
        __call__ method: Core operation of the Resampler class.

        """
        return resample_poly(data, self._up, self._down)

class AutoResampler(Resampler):

    """
    User-friendly version of the resampler. 

    Parameters
    ----------
        input_rate : int
        target_rate : int
        data : ndarray

    Returns
    -------
    ndarray
    """
    def __init__(self, input_rate, target_rate):

        """
        __init__ method: Core operation of the AutoResampler class.

        """
        if input_rate <= 0 or target_rate <= 0:
            raise ValueError("Sample rates must be positive.")

        self.input_rate = input_rate
        self.target_rate = target_rate
        self.gcd = np.gcd(int(input_rate), int(target_rate))
        self.up = int(target_rate // self.gcd)
        self.down = int(input_rate // self.gcd)

        super().__init__(self.up, self.down)
        
### MISC ###


class EnvelopeDetector(GenericModule):

    """
    EnvelopeDetector class: Extracts the amplitude envelope from a complex signal.

    Uses the Hilbert transform to compute the analytic signal and extracts the 
    amplitude envelope. This is useful for signal detection and monitoring where 
    amplitude variations indicate the presence of a signal.

    Parameters
    ----------
        N : int, optional
            Number of Fourier components. Default is None.
        axis : int, optional
            Axis along which to perform the transformation. Default is -1.
        data : ndarray

    Returns
    -------
    ndarray
        Amplitude envelope of the input signal.

    Notes
    -----
    Adding NoiseSuppressor after this block is highly recommended.
    """
    def __init__(self, N=None, axis=-1):

        """
        __init__ method: Core operation of the class.

        """
        self.N = N
        self.ax = axis
        pass

    def __call__(self, data):

        """
        __call__ method: Core operation of the EnvelopeDetector class.

        """
        analytic_signal = hilbert(data, self.N, self.ax)
        envelope = np.abs(analytic_signal)
        return envelope


class NoiseSuppressor(GenericModule):

    """
    Parameters
    ----------
        data : ndarray
        kernel : int
            kernel size of the filter.

    Returns
    -------
    ndarray
    """
    def __init__(self, kernel=3):

        """
        __init__ method: Core operation of the class.

        """
        self.kernel=kernel
        pass

    def __call__(self, data):

        """
        __call__ method: Core operation of the NoiseSuppressor class.

        """
        filtered = medfilt(data, kernel_size=self.kernel)
        return filtered


class DecisionMaker(GenericSink):

    """
    Decision maker, specifically for the peak detection pipe.

    Parameters
    ----------
        envelope : ndarray
            Calculate it with envelope detector first
        snr : float
            SNR

    Returns
    -------
    Bool
    """
    def __init__(self):

        """
        __init__ method: Core operation of the class.

        """
        pass
    
    def __call__(self, envelope, snr):

        """
        __call__ method: Core operation of the DecisionMaker class.

        """
        if snr > 10 and np.max(envelope) > 0.5:
            logging.info("Signal detected: high SNR and strong envelope")
            return True
        logging.info("Signal not detected")
        return False
    
    def flush(self):
        raise NotImplementedError("Not implemented yet.")


class FrequencyShift(GenericModule):

    """
    FrequencyShift

    Parameters
    ----------
        freq_shift_hz : int
        sample_rate : int
        data : ndarray

    Returns
    -------
    ndarray
    """
    def __init__(self, freq_shift_hz, sample_rate):

        """
        __init__ method: Core operation of the FrequencyShift class.

        """
        self.freq_shift_hz = freq_shift_hz
        self.sample_rate = sample_rate
        self.sample_index = 0

    def __call__(self, data):

        """
        __call__ method: Core operation of the FrequencyShift class.

        """
        t = np.arange(len(data)) / self.sample_rate
        shift = np.exp(-2j * np.pi * self.freq_shift_hz * (self.sample_index + t))
        self.sample_index += len(data)
        return data * shift

### SIGNAL MEASUREMENTS ###

class SignalNoiseRatioFFT(GenericModule):
    """
    SignalNoiseRatioFFT class: Calculates SNR from the power spectrum.

    This class uses FFT data to calculate the signal-to-noise ratio in dB.

    Parameters
    ----------
    fft_size : int
        Number of FFT points (default: 1024).
    noise_floor : float
        Minimum threshold to consider as noise (default: -100 dBm).
    """
    def __init__(self, fft_size=1024, noise_floor=-100.0):
        self.fft_size = fft_size
        self.noise_floor = noise_floor

    def __call__(self, data):
        """
        Calculates SNR from the input signal.

        Parameters
        ----------
        data : ndarray
            Time-domain signal data.

        Returns
        -------
        float
            Signal-to-noise ratio in dB.
        """
        #TODO: rewrite this in holy C

        spectrum = np.fft.fft(data, self.fft_size)
    
        power_spectrum = np.abs(spectrum)**2 / len(spectrum)
   
        signal_power = np.max(power_spectrum)  
        noise_power = np.mean(power_spectrum[power_spectrum < self.noise_floor])
        
        if noise_power <= 0:
            noise_power = 1e-12  

        
        snr = 10 * np.log10(signal_power / noise_power)
        logging.info(f"Calculated SNR: {snr} dB")
        return snr

class SignalPowerFFT(GenericModule):
    """
    SignalPowerFFT class: Calculates the power spectrum in dBm from FFT data.

    This class is used to transform time-domain signals into frequency-domain power levels.

    Parameters
    ----------
    reference_pwr : float
        Power of the noise floor (default: -90 dBm).
    fft_size : int
        Number of FFT points (default: 1024).
    """
    def __init__(self, reference_pwr=-90.0, fft_size=1024):
        self.ref_p = reference_pwr
        self.fft_size = fft_size

    def __call__(self, data):
        """
        Calculates the power spectrum in dBm from the input signal.

        Parameters
        ----------
        data : ndarray
            Time-domain signal data.

        Returns
        -------
        ndarray
            Power spectrum in dBm for each frequency bin.
        """
        #TODO: Split these guys into separate blocks 
        # Fourier <3
        spectrum = np.fft.fft(data, self.fft_size)
        
        #Magnitude
        power_spectrum = np.abs(spectrum)**2 / len(spectrum)
        
        #Converting to log scale 
        dBm_spectrum = 10 * np.log10(power_spectrum) + self.ref_p
        logging.info("Signal Power Spectrum (dBm): Computed")
        return dBm_spectrum




### AUDIO ###

class AudioNormalize(GenericModule):

    """
    Normalizes amplitudes of the audio.

    Parameters
    ----------
        data : ndarray

    Returns
    -------
    ndarray
    """
    def __init__(self):

        """
        __init__ method: Core operation of the class.

        """
        pass

    def __call__(self, data):

        """
        __call__ method: Core operation of the class.

        """
        data /= np.max(np.abs(data))
        return data

### DECODERS ###

class DecoderAPT(APT):

    """
    DecoderAPT class: Module for decoding APT images.

    Parameters
    ----------
        in_file : None or String
            just in case you need to read signal from file

        samplerate : int
            only suports 20800 for now
        
        data : ndarray
            Data to be decoded (audio).

    Returns
    -------
    ndarray

    Notes
    -----
    Dependent on the apt_decoder.
    """
    RATE = 20800
    NOAA_LINE_LENGTH = 2080

    def __init__(self, in_file=None, samplerate=RATE):

        """
        __init__ method: Core operation of the DecoderAPT class.

        """
        self._in = in_file
        self._signal = []

        if samplerate != self.RATE:
            raise Exception("Resample audio to {}".format(self.RATE))

        if self._in is not None:
            super().__init__(self._in)
         
    def __call__(self, data):

        """
        __call__ method: Core operation of the DecoderAPT class.

        """
        self._signal.append(data)
        
    def decode(self, outfile=None, imgshow=False):

        """
        decode method: Core operation of the DecoderAPT class.

        """
        if self._in is None:
            all_signal = np.concatenate(self._signal)
            truncate = self.RATE * int(len(all_signal)//self.RATE)
            if truncate > 0:
                all_signal = all_signal[:truncate]
            return super().decode(signal=all_signal, outfile=outfile, imgshow=imgshow)
        return super().decode(outfile=outfile, imgshow=imgshow)


### OUTPUT (SINKS) ###

class FileSink(GenericSink):

    """
    FileSink class: Module for saving data to the file

    Parameters
    ----------
        path

    Returns
    -------
    None

    Notes
    -----
    Do not use this to save audio/signal.
    """
    def __init__(self, path):

        """
        __init__ method: Core operation of the FileSink class.

        """
        self.path = path
        self.buffer = []

    def __call__(self, data):

        """
        __call__ method: Core operation of the FileSink class.

        """
        self.buffer.append(data)

    def flush(self):

        """
        flush method: Core operation of the FileSink class.

        """
        with open(self.path, 'wb') as f:
            for block in self.buffer:
                f.write(block.tobytes())

class FileAudioSink(GenericSink):

    """
    Writes audio to .wav

    Parameters
    ----------
        sample_rate : int

    Returns
    -------
    None
    """
    def __init__(self, filename, sample_rate=48000):

        """
        __init__ method: Core operation of the FileAudioSink class.

        """
        self.filename = filename
        self.sample_rate = sample_rate
        self.buffer = []

    def __call__(self, data):

        """
        __call__ method: Core operation of the FileAudioSink class.

        """
        self.buffer.append(data)

    def flush(self):

        """
        flush method: Core operation of the FileAudioSink class.

        """
        full = np.concatenate(self.buffer)
        full = (full / np.max(np.abs(full)) * 32767).astype(np.int16)
        with wave.open(self.filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(full.tobytes())

class StreamAudioSink(GenericSink):

    """
    Plays audio. Reatime.
    
    Parameters
    ----------
        sample_rate : int

    Returns
    -------
    None

    Notes
    -----
    Use flush() to stop.
    """
    def __init__(self, sample_rate=48000):

        """
        __init__ method: Core operation of the StreamAudioSink class.

        """
        self.sample_rate = sample_rate
        self.stream = sd.OutputStream(samplerate=sample_rate, channels=1)
        self.stream.start()

    def __call__(self, data):

        """
        __call__ method: Core operation of the StreamAudioSink class.

        """
        self.stream.write(data.astype(np.float32))

    def flush(self):

        """
        flush method: Core operation of the StreamAudioSink class.

        """
        self.stream.stop()
        self.stream.close()

class BufferAudioSink(GenericSink):

    """
    Plays audio. Non-realtime.

    Parameters
    ----------
        sample_rate : int

    Returns
    -------
    None
    """
    def __init__(self, sample_rate=48000):

        """
        __init__ method: Core operation of the BufferAudioSink class.

        """
        self.sample_rate = sample_rate
        self.buffer = []

    def __call__(self, data):

        """
        __call__ method: Core operation of the BufferAudioSink class.

        """
        self.buffer.append(data)

    def flush(self):

        """
        flush method: Core operation of the BufferAudioSink class.

        """
        all_data = np.concatenate(self.buffer)
        sd.play(all_data, self.sample_rate)
        sd.wait()