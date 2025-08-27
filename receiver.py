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
    def receive(self, freq, bw, output):
        pass
    
    @abstractmethod
    def set_gain(self, lna=16, vga=16, amp=False):
        pass

    @abstractmethod
    def stop(self):
        pass

class HackRFReceiver(ReceiverBase):
    def __init__(self):
        if not hackrflib_found:
            raise RuntimeError("HackRF library not found")
        self.sdr = HackRF()
        print(f"HackRF found, id: {self.sdr.get_serial_no()}")

        self.max_lna = 40
        self.max_vga = 62
    
    # [ DIRECT RADIO CONTROL, ONLY FOR DEBUGGING PURPOSES ]
    def get_dev_handle(self):
        return self.sdr

    def __check_gain_limits(self, value, max_val, lable="LNA"):
        if value < 0:
            value = 0
            print(f"{lable} is out of bonds, setting to {value} dB")
        elif value > max_val:
            value = max_val
            print(f"{lable} is out of bonds, setting to {value} dB")

        return value
        
    def set_gain(self, lna=16, vga=16, amp=False):
        self.lna = self.__check_gain_limits(lna, self.max_lna)
        self.vga = self.__check_gain_limits(vga, self.max_vga, lable="VGA")
        self.amp = amp

        self.sdr.lna_gain = self.lna
        self.sdr.vga_gain = self.vga
        self.sdr.amplifier_on = self.amp

        print(f"HackRF gain parameters updated:\n - LNA: {self.lna}\n - VGA: {self.vga}\n - AMP ON: {self.amp}")

    def receive(self, freq, bw, output):
        self.sdr.center_freq = freq
        self.sdr.sample_rate = bw

        print(f"Receiving from HackRF at {round((self.sdr.center_freq/1000000), 4)} MHz, BW: {round((self.sdr.sample_rate/1000000), 4)} MHz")
        
        self.sdr.start_rx(pipe_function=output)

    def stop(self):
        print(f"RX was manually stopped")
        self.sdr.stop_rx()

class RTLSdrReceiver(ReceiverBase):
    def __init__(self):
        if not rtlsdrlib_found:
            raise RuntimeError("RTL-SDR library not found")
        self.sdr = RtlSdr()
        print("RTL-SDR initialized")

    def receive(self, freq, bw, output):
        print(f"Receiving from RTL-SDR at {round((freq/1000000), 4)} MHz, BW: {round((bw/1000000), 4)} MHz")
        #TODO
    
    def set_gain(self, lna=0, **kwargs):
        #TODO
        pass

    def stop(self):
        #TODO
        pass

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


