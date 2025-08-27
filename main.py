import receiver
from receiver import Receiver
import numpy as np
import math
from time import sleep

from pylab import *

# TEST CODE 

def pipe(data: bytes) -> bool:
    a = np.array(data).astype(np.int8).astype(np.float64).view(np.complex128)
    strength = np.sum(np.absolute(a)) / len(a)
    dbfs = 20 * math.log10(strength)
    print(f"dBFS: { dbfs }")
    return False    # pipe function may return True to stop rx immediately

receiver = Receiver(
    receiver_type="rtl-sdr",   # "hackrf" / "rtl-sdr" / "auto"
    lna=20,                 # Only for HackRF
    vga=30,
    amp=True
)

receiver.configure(
    freq=98.5e6,            # 98.5 MHz
    bw=2e6,                 # 2 MHz bandwidth
    samples_num=2048       # or use pipe=process_samples
)


receiver.configure(pipe=pipe)
receiver.receive()
time.sleep(1)
recevier.stop()

samples = receiver.receive()

psd(samples, NFFT=1024, Fs=2e6/1e6, Fc=98.5e6/1e6)
xlabel('Frequency (MHz)')
ylabel('Relative power (dB)')
show()

