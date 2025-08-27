import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

try:
    from hackrf import HackRF
    hackrflib_found = True
except ImportError:
    hackrflib_found = False

try:
    from rtlsdr import RtlSdr
    rtlsdrlib_found = True
except ImportError:
    rtlsdrlib_found = False

# Constants for hardware gain limits
HACKRF_MAX_LNA_GAIN = 40
HACKRF_MAX_VGA_GAIN = 62


class ReceiverBase(ABC):
    def set_frequency(self, freq: int):
        self._set_freq(freq)

    def set_sample_rate(self, rate: int):
        self._set_sample_rate(rate)

    def receive_samples(self, freq: int, rate: int, samples_num: int):
        self.set_frequency(freq)
        self.set_sample_rate(rate)
        logger.info(f"Receiving {samples_num} samples at {rate/1e6:.2f} MHz from {freq/1e6:.2f} MHz")
        return self._read_samples(samples_num)

    def receive_stream(self, freq: int, rate: int, pipe):
        try:
            self.set_frequency(freq)
            self.set_sample_rate(rate)
            logger.info(f"Streaming from {freq/1e6:.2f} MHz at {rate/1e6:.2f} MHz")
            self._start_stream(pipe)
        except NotImplementedError as e:
            logger.warning(e)

    def stop(self):
        try:
            self._stop_stream()
            logger.info("RX stopped")
        except NotImplementedError as e:
            logger.warning(e)

    def set_gain(self, lna, vga, amp):
        try:
            self._set_gain(lna, vga, amp)
        except NotImplementedError as e:
            logger.warning(e)

    @abstractmethod
    def _set_freq(self, freq): pass

    @abstractmethod
    def _set_sample_rate(self, rate): pass

    @abstractmethod
    def _read_samples(self, num): pass

    @abstractmethod
    def _start_stream(self, pipe): pass

    @abstractmethod
    def _stop_stream(self): pass

    @abstractmethod
    def _set_gain(self, **kwargs): pass


class HackRFReceiver(ReceiverBase):
    def __init__(self):
        self.sdr = HackRF()

    def _set_freq(self, freq):
        self.sdr.center_freq = freq

    def _set_sample_rate(self, rate):
        self.sdr.sample_rate = rate

    def _read_samples(self, num):
        return self.sdr.read_samples(num)

    def _start_stream(self, pipe):
        self.sdr.start_rx(pipe_function=pipe)

    def _stop_stream(self):
        self.sdr.stop_rx()

    def _set_gain(self, lna=16, vga=16, amp=False):
        if lna > HACKRF_MAX_LNA_GAIN:
            logger.warning(f"LNA gain exceeds max ({HACKRF_MAX_LNA_GAIN}), clipping to max")
        if vga > HACKRF_MAX_VGA_GAIN:
            logger.warning(f"VGA gain exceeds max ({HACKRF_MAX_VGA_GAIN}), clipping to max")
        self.sdr.lna_gain = min(max(0, lna), HACKRF_MAX_LNA_GAIN)
        self.sdr.vga_gain = min(max(0, vga), HACKRF_MAX_VGA_GAIN)
        self.sdr.amplifier_on = amp

# Will not work for now. RTL-SDR will not be supoprted in first release
class RtlSdrReceiver(ReceiverBase):
    def __init__(self):
        self.sdr = RtlSdr()

    def _set_freq(self, freq):
        self.sdr.center_freq = freq

    def _set_sample_rate(self, rate):
        self.sdr.sample_rate = rate

    def _read_samples(self, num):
        return self.sdr.read_samples(num)

    def _start_stream(self, pipe):
        raise NotImplementedError("Stream RX is not supported by RTL-SDR")

    def _stop_stream(self):
        raise NotImplementedError("Stream RX is not supported by RTL-SDR")

    def _set_gain(self, lna="auto", **_):
        self.sdr.gain = lna


class ReceiverFactory:
    @staticmethod
    def create(receiver_type="auto") -> ReceiverBase:
        if receiver_type == "hackrf" and hackrflib_found:
            return HackRFReceiver()
        elif receiver_type == "rtl-sdr" and rtlsdrlib_found:
            return RtlSdrReceiver()
        elif receiver_type == "auto":
            if hackrflib_found:
                return HackRFReceiver()
            elif rtlsdrlib_found:
                return RtlSdrReceiver()
        raise RuntimeError("No compatible SDR device found")


class Receiver:
    def __init__(self, 
                 receiver_type="auto", 
                 freq=None, bw=None, 
                 samples_num=0, 
                 pipe=None, 
                 amp=False, 
                 lna=0,
                 vga=0):
        self.device: ReceiverBase = ReceiverFactory.create(receiver_type)
        self.freq = freq
        self.bw = bw
        self.samples_num = samples_num
        self.pipe = pipe
        self.amp = amp
        self.lna = lna
        self.vga = vga

        logger.info(f"Receiver created: {self.device.__class__.__name__}")
        
        self.device.set_gain(self.lna, self.vga, self.amp)

    @property
    def sample_rate(self):
        return self.bw

    def set_gain(self, lna=None, vga=None, amp=None):
        if lna is None:
            lna = self.lna
        if vga is None:
            vga = self.vga
        if amp is None:
            amp = self.amp

        self.device.set_gain(lna, vga, amp)

    def set_freq(self, freq: int):
        if freq is None:
            raise ValueError("Frequency can't be None!")
        
        self.device.set_frequency(freq)

    def set_bandwith(self, bw: int):
        if bw is None:
            raise ValueError("Bandwith can't be None!")

        self.device.set_sample_rate(bw)

    def receive_stream(self):
        if self.freq is None or self.bw is None:
            raise ValueError("freq and bw must be set before receiving")

        return self.device.receive_stream(
            freq=self.freq,
            rate=self.bw,
            pipe=self.pipe,
        )
    
    def receive_samples(self, samples_num=0):
        if self.freq is None or self.bw is None:
            raise ValueError("freq and bw must be set before receiving")

        if samples_num == 0:
            samples_num = self.samples_num

        return self.device.receive_samples(
            freq=self.freq,
            rate=self.bw,
            samples_num=samples_num
        )

    def stop(self):
        self.device.stop()
