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

class ReceiverBase(ABC):
    def receive(self, freq, bw, pipe=None, samples_num=0):
        self._set_freq(freq)
        self._set_sample_rate(bw)
        logger.info(f"Receiving: {freq / 1e6:.2f} MHz @ {bw / 1e6:.2f} MHz")
        if samples_num is not None and samples_num > 0:
            return self._read_samples(samples_num)
        elif pipe is not None:
            self._start_stream(pipe)
        else:
            raise ValueError("Provide samples_num or pipe")

    def stop(self):
        self._stop_stream()
        logger.info("RX stopped")

    def set_gain(self, **kwargs):
        try:
            self._set_gain(**kwargs)
        except NotImplementedError:
            logger.warning("Gain control not supported by this device")

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
        if lna > 40: logging.warning(f"Maximum LNA is 40, got {lna}")
        if vga > 62: logging.warning(f"Maximum VGA is 62, got {vga}")

        self.sdr.lna_gain = min(max(0, lna), 40)
        self.sdr.vga_gain = min(max(0, vga), 62)
        self.sdr.amplifier_on = amp


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
        raise NotImplementedError(f"Stream RX is not supported by RTL-SDR")

    def _stop_stream(self):
        raise NotImplementedError(f"Stream RX is not supported by RTL-SDR")

    def _set_gain(self, lna="auto", **kwargs):
        self.sdr.gain = lna


class ReceiverFactory:
    @staticmethod
    def create(receiver_type="auto"):
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
    def __init__(self, receiver_type="auto", **kwargs):
        self.device: ReceiverBase = ReceiverFactory.create(receiver_type)
        self.config = {
            "freq": None,
            "bw": None,
            "samples_num": 0,
            "pipe": None
        }
        self.config.update(kwargs)
        logger.info(f"Receiver created: {self.device.__class__.__name__}")
        gain_args = {k: v for k, v in kwargs.items() if k in ('lna', 'vga', 'amp')}
        if gain_args:
            self.device.set_gain(**gain_args)

    def configure(self, freq=None, bw=None, samples_num=None, pipe=None):
        if freq is not None:
            self.config["freq"] = freq
        if bw is not None:
            self.config["bw"] = bw
        if samples_num is not None:
            self.config["samples_num"] = samples_num
        if pipe is not None:
            self.config["pipe"] = pipe

    def receive_stream(self, pipe=None):
        if pipe == None:
            pipe = self.config["pipe"]
        
        #receive continiously
        return self.device.receive(
            self.config["freq"],
            self.config["bw"],
            pipe,
            None
        )
    
    def receive_samples(self, samples_num=None):
        if samples_num == None:
            samples_num = self.config["samples_num"]

        #send None as pipe to set receiver into limited mode
        return self.device.receive(
            self.config["freq"],
            self.config["bw"],
            None,
            samples_num
        )

    def stop(self):
        self.device.stop()

