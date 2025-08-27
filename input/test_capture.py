import pytest
import numpy as np
from pathlib import Path
from scipy.io.wavfile import write
from capture import BasebandFileReader


@pytest.fixture
def dummy_wav(tmp_path):
    sample_rate = 8000
    duration = 1
    samples = (np.random.rand(sample_rate * duration) * 32767).astype(np.int16)
    wav_path = tmp_path / "dummy.wav"
    write(wav_path, sample_rate, samples)
    return wav_path, samples, sample_rate


def test_basebandfile_reader_initialization(dummy_wav):
    wav_path, original_samples, sample_rate = dummy_wav
    reader = BasebandFileReader(str(wav_path), bits=16)

    assert reader.sample_rate == sample_rate
    assert reader.samples_available == len(original_samples)
    assert isinstance(reader.data, np.ndarray)
    assert reader.data.dtype == np.float32
    assert np.all(reader.data >= -1.0) and np.all(reader.data <= 1.0)


def test_receive_samples_returns_correct_length(dummy_wav):
    wav_path, _, _ = dummy_wav
    reader = BasebandFileReader(str(wav_path), bits=16)
    samples = reader.receive_samples(100)

    assert isinstance(samples, np.ndarray)
    assert len(samples) == 100


def test_receive_stream_callback_invoked_correctly(dummy_wav):
    wav_path, _, _ = dummy_wav
    reader = BasebandFileReader(str(wav_path), bits=16)

    chunks = []

    def callback(chunk):
        chunks.append(chunk)

    reader.receive_stream(callback, chunk_size=256)

    assert len(chunks) > 0
    assert all(isinstance(chunk, np.ndarray) for chunk in chunks)
    assert sum(len(c) for c in chunks) <= len(reader.data)


def test_stop_function_interrupts_stream(dummy_wav):
    wav_path, _, _ = dummy_wav
    reader = BasebandFileReader(str(wav_path), bits=16)

    calls = []

    def callback(chunk):
        calls.append(chunk)
        reader.stop()

    reader.receive_stream(callback, chunk_size=128)

    assert len(calls) == 1  # should stop after first callback


def test_reset_resets_position(dummy_wav):
    wav_path, _, _ = dummy_wav
    reader = BasebandFileReader(str(wav_path), bits=16)
    _ = reader.receive_samples(100)
    assert reader.position == 100

    reader.reset()
    assert reader.position == 0
