import receiver
from receiver import *
import numpy as np
import math
from time import sleep

from pylab import *

def pipe(data: bytes) -> bool:
    a = np.array(data).astype(np.int8).astype(np.float64).view(np.complex128)
    strength = np.sum(np.absolute(a)) / len(a)
    dbfs = 20 * math.log10(strength)
    print(f"dBFS: { dbfs }")
    return False    # pipe function may return True to stop rx immediately


receiver = ReceiverFactory.create()
receiver.set_gain(20, 41, True)
receiver.receive(100e6, 2e6, pipe)
sleep(1)
receiver.stop()

sdr = receiver.get_dev_handle()

samples = sdr.read_samples(2e6)

psd(samples, NFFT=1024, Fs=sdr.sample_rate/1e6, Fc=sdr.center_freq/1e6)
xlabel('Frequency (MHz)')
ylabel('Relative power (dB)')
show()

