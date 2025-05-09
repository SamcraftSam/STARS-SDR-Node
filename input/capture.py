import numpy as np
import soundfile as sf
from scipy.io import wavfile
import re

BIT_DEPTH_TO_DTYPE = {
    8: np.uint8,
    16: np.int16,
    32: np.int32,
}

class BasebandFileReader:
    def __init__(self, filename, bits=16, samples_num=5016):
        self._filename = filename
        self._bits = bits
        self._samples_num = samples_num
        self._pos = 0
        self._stop_call = False

        try:
            self._dtype = BIT_DEPTH_TO_DTYPE[bits]
        except KeyError:
            raise ValueError(f"Unsupported bit depth: {bits}")

        self._samplerate, raw_data = wavfile.read(self._filename)

        if np.issubdtype(raw_data.dtype, np.integer):
            scale = float(2**(bits - 1))
            self._data = raw_data.astype(np.float32) / scale
        elif np.issubdtype(raw_data.dtype, np.floating):
            self._data = raw_data.astype(np.float32)
        else:
            raise ValueError("Unsupported WAV data type")
        
        match = re.search(r'baseband_(\d+)Hz', self.filename)

        self._frequency = None
        if match:
            self._frequency = int(match.group(1))
            

    @property
    def position(self):
        return self._pos

    @property
    def sample_rate(self):
        return self._samplerate

    @property
    def samples_available(self):
        return len(self._data) - self._pos

    @property
    def data(self):
        return self._data.copy()

    @property
    def frequency(self):
        return self._frequency

    def receive_stream(self, callback, chunk_size=1024):
        self._stop_call = False
        chunk_counter = 0

        while self._pos < len(self._data) and not self._stop_call:
            chunk = self._data[self._pos:self._pos + chunk_size]
            callback(chunk.copy())
            self._pos += chunk_size
            chunk_counter += 1

    def receive_samples(self, samplesnum=0):
        num_samples = self._samples_num if samplesnum == 0 else samplesnum
        samples = self._data[self._pos:self._pos + num_samples].copy()
        self._pos += num_samples
        return samples

    def stop(self):
        self._stop_call = True

    def reset(self):
        self._pos = 0
        self._stop_call = False

class StreamedFileReader:
    """
    StreamedFileReader: Pseudo-realtime reader for large audio files.
    
    Mimics the interface of BasebandFileReader for seamless integration.
    Uses streaming to avoid loading entire file into memory.

    Parameters
    ----------
    filename : str
        Path to the audio file.
    bits : int
        Bit depth of the audio file (8, 16, 32).
    samples_num : int
        Number of samples per chunk (default: 5016).
    """
    def __init__(self, filename, bits=16, samples_num=5016):
        self._filename = filename
        self._bits = bits
        self._samples_num = samples_num
        self._pos = 0
        self._stop_call = False

        try:
            self._dtype = BIT_DEPTH_TO_DTYPE[bits]
        except KeyError:
            raise ValueError(f"Unsupported bit depth: {bits}")

        try:
            # Open audio file in streaming mode
            self._file = sf.SoundFile(self._filename, mode='r')
            self._samplerate = self._file.samplerate
        except Exception as e:
            raise IOError(f"Error opening file: {e}")

        # Extract frequency from the filename
        match = re.search(r'baseband_(\d+)Hz', filename)
        self._frequency = int(match.group(1)) if match else None

    @property
    def position(self):
        return self._pos

    @property
    def sample_rate(self):
        return self._samplerate

    @property
    def samples_available(self):
        return len(self._file) - self._pos

    @property
    def data(self):
        """
        Reads the entire file content (if necessary).
        """
        self._file.seek(0)
        data = self._file.read(dtype=self._dtype)
        self._file.seek(self._pos)
        return data

    @property
    def frequency(self):
        return self._frequency

    def receive_stream(self, callback, chunk_size=1024):
        """
        Pseudo-realtime streaming of the audio file.
        """
        self._stop_call = False

        while not self._stop_call:
            chunk = self._file.read(chunk_size, dtype=self._dtype)

            if len(chunk) == 0:
                break

            # Normalize if necessary
            if np.issubdtype(chunk.dtype, np.integer):
                scale = float(2 ** (self._bits - 1))
                chunk = chunk.astype(np.float32) / scale

            _not_a_stop_call = callback(chunk.copy())
            # if not _not_a_stop_call:
            #     self._stop_call = True
            self._pos += len(chunk)

    def receive_samples(self, samples_num=0):
        """
        Get a specified number of samples from the file.
        """
        num_samples = self._samples_num if samples_num == 0 else samples_num
        chunk = self._file.read(num_samples, dtype=self._dtype)

        if len(chunk) == 0:
            self._stop_call = True
            return np.array([])

        # Normalize if necessary
        if np.issubdtype(chunk.dtype, np.integer):
            scale = float(2 ** (self._bits - 1))
            chunk = chunk.astype(np.float32) / scale

        self._pos += len(chunk)
        return chunk

    def stop(self):
        """
        Stop the streaming process.
        """
        self._stop_call = True

    def reset(self):
        """
        Reset the file stream position.
        """
        self._file.seek(0)
        self._pos = 0
        self._stop_call = False

    def close(self):
        """
        Close the file when done.
        """
        self._file.close()

    def __del__(self):
        """
        Ensure the file is closed on object deletion.
        """
        if not self._file.closed:
            self._file.close()
