import logging
import numpy as np


from file_processor import BasebandFileReader
from receiver import Receiver
from time import sleep

from pylab import *


logging.basicConfig(level=logging.INFO)

# TEST CODE 

freq = 100e6

def pipe(data: bytes) -> bool:
    iq = np.frombuffer(data, dtype=np.int8).astype(np.float32)
    
    if len(iq) % 2 != 0:
        return False
    
    iq = iq[::2] + 1j * iq[1::2]
    
    # Estimate signal power (normalized)
    strength = np.sum(np.abs(iq) ** 2) / len(iq)
    
   
    reference_power_dBm = -90 
    dBm = 10 * np.log10(strength) + reference_power_dBm
    
    print(f"dBm: {dBm}")
    
    return False

# receiver = Receiver(
#     receiver_type="hackrf",   # "hackrf" / "rtl-sdr" / "auto"
#     lna=20,                 # Only for HackRF
#     vga=30,
#     amp=True
# )

# receiver.configure(
#     freq=freq,            
#     bw=20e6,                 
#     samples_num=5016,       # or use pipe=process_samples
#     pipe=pipe
# )

# receiver.receive_stream()
# time.sleep(1)
# receiver.stop()

# samples = receiver.receive_samples()

# psd(samples, NFFT=1024, Fs=20e6/1e6, Fc=freq/1e6)
# xlabel('Frequency (MHz)')
# ylabel('Relative power (dB)')
# show()


# file = BasebandFileReader("/home/alex/Downloads/baseband_914975000Hz_22-51-24_12-04-2025.wav")

file = BasebandFileReader("/home/alex/.config/sdrpp/recordings/baseband_137805000Hz_18-38-26_30-03-2025.wav")
# file.receive_stream(pipe)
# time.sleep(1)
# file.stop()

samples = file.receive_samples(100*1024)
print(samples)
iq = samples[:, 0] + 1j * samples[:, 1]
print(iq)
psd(iq, NFFT=1024, Fs=20e6/1e6, Fc=137.805)
show()

