import logging
import numpy as np

from signal_processing import *
from file_processor import BasebandFileReader
from receiver import Receiver
from time import sleep

from pylab import *


logging.basicConfig(level=logging.INFO)

# TEST CODE 

freq = 100e6

def pipe_old(data: bytes) -> bool:
    print(type(data))
    iq = np.frombuffer(data, dtype=np.int8)#.astype(np.float32)
    print(type(iq[0]))
    if len(iq) % 2 != 0:
        return False
    
    iq = iq[::2] + 1j * iq[1::2]
    
    # Estimate signal power (normalized)
    strength = np.sum(np.abs(iq) ** 2) / len(iq)
    
   
    reference_power_dBm = -90 
    dBm = 10 * np.log10(strength) + reference_power_dBm
    
    print(f"dBm: {dBm}")

    return False

audio = AudioSink(44000)

builder = RadioPipelineBuilder()
builder.add_step(BytesToComplex(normalize=True))
builder.add_step(FrequencyShift(137.625e6-137.805e6-40000, 20e6))
builder.add_step(BandPassFilterFIR(20000, 40000, 20e6))
builder.add_step(FMQuadratureDemod())
builder.add_step(audio)
pipe_apt = builder.build()

def pipe_apt_full(data: bytes):
    pipe_apt(data)
    logging.warn("Playing the chunk!")
    


# receiver = Receiver(
#     receiver_type="hackrf",   # "hackrf" / "rtl-sdr" / "auto"
#     lna=10,                 # Only for HackRF
#     vga=15,
#     amp=True,
#     freq=freq,            
#     bw=20e6,                 
#     samples_num=5016,       # or use pipe=process_samples
#     pipe=pipe
# )


# receiver.receive_stream()
# time.sleep(1)
# receiver.stop()

# samples = receiver.receive_samples()
# print(type(samples[0]))

# psd(samples, NFFT=1024, Fs=20e6/1e6, Fc=freq/1e6)
# xlabel('Frequency (MHz)')
# ylabel('Relative power (dB)')
# show()


# file = BasebandFileReader("/home/alex/Downloads/baseband_914975000Hz_22-51-24_12-04-2025.wav")

file = BasebandFileReader("/home/alex/.config/sdrpp/recordings/baseband_137805000Hz_18-38-26_30-03-2025.wav")
file.receive_stream(pipe_apt_full, 1024*1000)
# time.sleep(1)
# file.stop()

# samples = file.receive_samples(1024*1000)

# #print(pipe(samples))

# print(type(samples[0]))
# print(samples)
# iq = samples[:, 0] + 1j * samples[:, 1]
# print(type(iq[0]))
# print(iq)
# psd(iq, NFFT=1024, Fs=20e6/1e6, Fc=0)
# show()

