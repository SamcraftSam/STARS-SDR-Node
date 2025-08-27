import logging
import numpy as np

from mqtt.sender import *
from pipelines.pipelines import *
from pipelines.signal_scanner import *
from apt_tools.apt_colorize import APTColorizer2D
from input.capture import StreamedFileReader
from input.receiver import Receiver
from time import sleep

from pylab import *


logging.basicConfig(level=logging.INFO)

publisher = MQTTImagePublisher("broker.hivemq.com")
encoder = ImageEncoder()
service = APTMQTTService(publisher, encoder)

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
file = StreamedFileReader(filename="/home/alex/.config/sdrpp/recordings/baseband_137805000Hz_18-38-26_30-03-2025.wav")
#file = StreamedFileReader(filename="/home/alex/.config/sdrpp/recordings/baseband_137792229Hz_18-17-43_28-03-2025.wav")

print(f"File samplerate: {file.sample_rate/1e6} MHz")
#print(f"Shape: {file._data.shape}, type: {file._data.dtype}")
#print(f"Duration: {file._data.shape[0] / file.sample_rate:.2f} seconds")

constructor = DSPipelines()
decoder = DecoderAPT()
aptcolor = APTColorizer2D()

detector = SignalDetectorPipeline(file.frequency)


def pipe_detector(data):
    if detector.run(data):
        file.stop()
        return False
    return True

#block until we find NOAA freq
file.receive_stream(pipe_detector)

shift = abs(file.frequency - 137630000)#detector._best_freq)
print(f"Shift is: {shift}")
pipe_apt = constructor.construct_noaa_apt_decoder(file, decoder, -180000)
print(f"Type of pipe_apt: {type(pipe_apt)}")
print(f"Callable: {callable(pipe_apt)}")

print("Receiving stream....")
file.receive_stream(pipe_apt, 1024*20)

#pipe_apt(file.receive_samples(1024*1024))

print(f"Decoder buffer size: {sum(len(s) for s in decoder._signal)}")
print("Decoding...")
data = decoder.decode(outfile="images_local/test_auto.png")

aptcolor.colorize(data=data, outfile="images_local/test_colorized_auto.png", show=True)


#publisher.connect()

#service.send_image("images_local/test_colorized_auto.png", "NOAA-19", "Berlin, Germany", coordinates=(52.52, 13.40))

#publisher.disconnect()
