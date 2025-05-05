import pytest
import numpy as np
from unittest.mock import MagicMock
from pathlib import Path
from capture import BasebandFileReader
from receiver import Receiver, ReceiverFactory, ReceiverBase


# === Test for capture.py ===

def test_baseband_reader_loads_correctly(tmp_path):
    # Create dummy wav file
    from scipy.io.wavfile import write
    rate = 44100
    duration = 1
    samples = (np.random.rand(rate * duration) * 32767).astype(np.int16)
    wav_path = tmp_path / "test.wav"
    write(wav_path, rate, samples)

    reader = BasebandFileReader(str(wav_path), bits=16)
    assert reader.sample_rate == rate
    assert reader.samples_available == len(samples)
    assert isinstance(reader.receive_samples(100), np.ndarray)


def test_baseband_reader_stream_callback(tmp_path):
    from scipy.io.wavfile import write
    rate = 1000
    samples = (np.ones(rate) * 128).astype(np.int16)
    wav_path = tmp_path / "dummy.wav"
    write(wav_path, rate, samples)

    reader = BasebandFileReader(str(wav_path), bits=16)
    chunks = []

    def cb(chunk):
        chunks.append(chunk)

    reader.receive_stream(cb, chunk_size=100)
    assert sum(len(c) for c in chunks) <= rate


class DummyReceiver:
    def __init__(self):
        self.freq = None
        self.sampling_rate = None
        self.bandwidth = None
        self.rate = None  # Added to satisfy test
        self.samples = []
        self.pipe_called = False
        self.lna = 0
        self.vga = 0
        self.amp = False

    def _set_gain(self, lna, vga, amp):
        self.lna = lna
        self.vga = vga
        self.amp = amp

    def set_gain(self, lna, vga, amp):
        self._set_gain(lna, vga, amp)

    def set_frequency(self, freq):
        self.freq = freq

    def set_bandwidth(self, bw):
        self.bandwidth = bw
        self.rate = bw  # Set both to satisfy expectations

    def set_sample_rate(self, bw):
        self.set_bandwidth(bw)

    def receive_samples(self, freq, rate, samples_num):
        self.freq = freq
        self.bandwidth = rate
        self.rate = rate
        self.samples = [complex(0, 0)] * samples_num
        return np.array(self.samples)

    def receive_stream(self, freq=None, rate=None, pipe=None):
        self.pipe_called = True
        self.freq = freq
        self.bandwidth = rate
        self.rate = rate
        if pipe:
            pipe([complex(0, 0)] * 10)

    def stop(self):
        self.pipe_called = False

def test_receiver_wrapper(monkeypatch):
    monkeypatch.setattr(ReceiverFactory, "create", lambda *_: DummyReceiver())
    r = Receiver(receiver_type="auto", freq=137e6, bw=20800, samples_num=10)
    r.set_gain(lna=10, vga=20, amp=True)
    r.set_freq(137e6)
    r.set_bandwith(20800)

    samples = r.receive_samples()
    assert isinstance(samples, np.ndarray)
    assert len(samples) == 10

    r.receive_stream()
    r.stop()

    # Check internal settings
    assert r.device.freq == 137e6
    assert r.device.rate == 20800
    assert r.device.pipe_called is False
