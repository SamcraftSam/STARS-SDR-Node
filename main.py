import logging
import numpy as np

from pipelines.pipelines import *
from apt_tools.apt_colorize import APTColorizer2D
from input.capture import BasebandFileReader
from input.receiver import Receiver
from time import sleep

from pylab import *


logging.basicConfig(level=logging.INFO)

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

print("Reading WAV...")

# true file
file = BasebandFileReader(filename="/home/alex/.config/sdrpp/recordings/baseband_137805000Hz_18-38-26_30-03-2025.wav")
#file = BasebandFileReader(filename="/home/alex/.config/sdrpp/recordings/baseband_137792229Hz_18-17-43_28-03-2025.wav")

print(f"File samplerate: {file.sample_rate/1e6} MHz")
print(f"Shape: {file._data.shape}, type: {file._data.dtype}")
print(f"Duration: {file._data.shape[0] / file.sample_rate:.2f} seconds")

constructor = DSPipelines()
decoder = DecoderAPT()
aptcolor = APTColorizer2D()

pipe_apt = constructor.construct_noaa_apt_decoder(file, decoder)

print("Receiving stream....")
file.receive_stream(pipe_apt, 1024*20)

#pipe_apt(file.receive_samples(1024*1024))

print(f"Decoder buffer size: {sum(len(s) for s in decoder._signal)}")
print("Decoding...")
data = decoder.decode(outfile="images/test_grey.png")

aptcolor.colorize(data=data, outfile="images/test_colorized.png", show=True)

