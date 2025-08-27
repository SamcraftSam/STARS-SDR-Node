from abc import ABC, abstractmethod

try:
    from pyhackrf2 import HackRF
    hackrflib_found = True
except ImportError as e:
    print(f"[WARNING] HackRF Library import error: {e}")
    hackrflib_found = False

try:
    from rtlsdr import RtlSdr
    rtlsdrlib_found = True
except ImportError as e:
    print(f"[WARNING] RTL-SDR Library import error: {e}")
    rtlsdrlib_found = False

class ReceiverBase(ABC):
    @abstractmethod
    def receive(self, freq, bw, output='buffer'):
        pass

class HackRFReceiver(ReceiverBase):
    def __init__(self):
        if not hackrflib_found:
            raise RuntimeError("HackRF library not found")
        self.sdr = HackRF()
        print(f"HackRF found, id: {self.sdr.get_serial_no()}")

    def receive(self, freq, bw, output='buffer'):
        print(f"Receiving from HackRF at {round((freq/1000000), 4)} MHz, BW: {round((bw/1000000), 4)} MHz")
        #TODO

class RTLSdrReceiver(ReceiverBase):
    def __init__(self):
        if not rtlsdrlib_found:
            raise RuntimeError("RTL-SDR library not found")
        self.sdr = RtlSdr()
        print("RTL-SDR initialized")

    def receive(self, freq, bw, output='buffer'):
        print(f"Receiving from RTL-SDR at {round((freq/1000000), 4)} MHz, BW: {round((bw/1000000), 4)} MHz")
        #TODO

class ReceiverFactory:
    @staticmethod
    def create(receiver_type="auto"):
        if receiver_type == "auto":
            if hackrflib_found:
                return HackRFReceiver()
            elif rtlsdrlib_found:
                return RTLSdrReceiver()
            else:
                raise RuntimeError("No SDR device found")
        elif receiver_type == "hackrf":
            return HackRFReceiver()
        elif receiver_type == "rtl-sdr":
            return RTLSdrReceiver()
        else:
            raise ValueError(f"Unknown receiver type: {receiver_type}")

# TEST
receiver = ReceiverFactory.create()
receiver.receive(100e6, 2e6)

