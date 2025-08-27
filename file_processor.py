import numpy as np
from scipy.io import wavfile

class BasebandFileReader:
    def __init__(self, filename, bits=16, samples_num=5016):
        self.filename = filename
        self.bits = bits
        self.samplerate, self.data = wavfile.read(self.filename)
        self.pos = 0
        self.scale = 32767 if self.bits == 16 else (2**(self.bits - 1) - 1)
        self.__stop_call = False


        ### For compatibility with receiver.py sampled mode reception
        self.samples_num = samples_num

    def receive_stream(self, callback, chunk_size=1024):
        self.__stop_call = False
        if self.data.dtype == np.float32 or self.data.dtype == np.float64:
            self.data = (self.data * self.scale).astype(np.int16)

        while self.pos < len(self.data) and not self.__stop_call:
            chunk = self.data[self.pos:self.pos + chunk_size]
            buf = chunk.astype(np.int16).tobytes()
            callback(buf)
            self.pos += chunk_size

    def receive_samples(self, samplesnum=0):
        if self.data.dtype == np.float32 or self.data.dtype == np.float64:
            self.data = (self.data * self.scale).astype(np.int16)

        ### For compatibility with receiver.py sampled mode reception
        if samplesnum == 0:
            return self.data[0:self.samples_num]
        
        return self.data[0:samplesnum]
    
    def stop(self):
        self.__stop_call = True
