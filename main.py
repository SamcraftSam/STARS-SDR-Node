import logging
import numpy as np

from signal_processing import *
from file_processor import BasebandFileReader
from receiver import Receiver
from time import sleep

from pylab import *

AUDIO_SAMPLE_RATE = 48000
RESAMPLED_AUDIO_SAMPLE_RATE = 20800


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

print("Reading WAV...")

file = BasebandFileReader(filename="/home/alex/.config/sdrpp/recordings/baseband_137805000Hz_18-38-26_30-03-2025.wav")

print(f"File samplerate: {file.sample_rate/1e6} MHz")
print(f"Shape: {file._data.shape}, type: {file._data.dtype}")
print(f"Duration: {file._data.shape[0] / file.sample_rate:.2f} seconds")

audio = FileAudioSink("sounds/demodulated4.3.wav", 
                      RESAMPLED_AUDIO_SAMPLE_RATE)

print("Created File sink")

decoder = DecoderAPT()

print("Created decoder")

# dc_test = DecoderAPT("sounds/demodulated4.2.wav")
# dc_test.decode(outfile="images/img1.png")

builder = RadioPipelineBuilder()
#builder.add_step(BytesToComplex(data_type=np.int16, normalize=True))
builder.add_step(WavToComplex(normalize=True))
builder.add_step(FrequencyShift(-180000, file.sample_rate))
builder.add_step(LowPassFilterIIR(20000, file.sample_rate))
builder.add_step(FMQuadratureDemod())
builder.add_step(AutoDecimator(file.sample_rate, AUDIO_SAMPLE_RATE))
#builder.add_step(AudioNormalize())
builder.add_step(BandPassFilterFIR(10, 5000, AUDIO_SAMPLE_RATE))
builder.add_step(AutoResampler(AUDIO_SAMPLE_RATE, RESAMPLED_AUDIO_SAMPLE_RATE))
#builder.add_step(audio)
builder.add_step(decoder)
pipe_apt = builder.build()

print("Pipe has been built")

def pipe_apt_full(data):
    #logging.warning("Playing the chunk!")
    pipe_apt(data)

print("Receiving stream....")
file.receive_stream(pipe_apt_full, 1024*20)

#pipe_apt_full(file.data)
#audio.flush()
print("Decoding...")
decoder.decode(outfile="sounds/out.png", imgshow=True)