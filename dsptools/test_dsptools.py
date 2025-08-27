import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))


import pytest
import numpy as np
from dsptools import *

@pytest.fixture
def dummy_stereo_signal():
    return np.stack([np.arange(10), np.arange(10) * -1], axis=1)

def test_wav_to_complex(dummy_stereo_signal):
    converter = WavToComplex()
    result = converter(dummy_stereo_signal)
    assert result.dtype == np.complex128
    assert len(result) == 10

def test_wav_to_complex_invalid():
    mono = np.arange(10).reshape(-1, 1)
    with pytest.raises(ValueError):
        WavToComplex()(mono)

def test_bytes_to_complex():
    iq_data = np.array([1, -1] * 10, dtype=np.int8).tobytes()
    mod = BytesToComplex(data_type=np.int8)
    out = mod(iq_data)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.complex64 or out.dtype == np.complex128
    assert len(out) == 10

def test_bytes_to_complex_odd_length():
    with pytest.raises(ValueError):
        BytesToComplex()(bytes([1, 2, 3]))

def test_lowpass_iir():
    filt = LowPassFilterIIR(1000, 48000)
    data = np.random.randn(1024)
    filtered = filt(data)
    assert len(filtered) == len(data)

def test_lowpass_fir():
    filt = LowPassFilterFIR(1000, 48000)
    data = np.random.randn(1024)
    filtered = filt(data)
    assert len(filtered) == len(data)

def test_bandpass_fir():
    filt = BandPassFilterFIR(500, 2000, 48000)
    data = np.random.randn(1024)
    filtered = filt(data)
    assert len(filtered) == len(data)

def test_fm_demod():
    sig = np.exp(1j * 2 * np.pi * np.arange(1000) * 0.01)
    demod = FMQuadratureDemod()
    out = demod(sig)
    assert len(out) == len(sig)

def test_decimator():
    sig = np.random.randn(1000)
    dec = Decimator(5)
    out = dec(sig)
    assert len(out) < len(sig)

def test_auto_decimator_warning(caplog):
    sig = np.random.randn(1000)
    dec = AutoDecimator(48000, 5000)
    out = dec(sig)
    assert len(out) < len(sig)

def test_resampler():
    sig = np.random.randn(100)
    res = Resampler(up=2, down=3)
    out = res(sig)
    assert isinstance(out, np.ndarray)

def test_auto_resampler():
    sig = np.random.randn(100)
    res = AutoResampler(48000, 32000)
    out = res(sig)
    assert isinstance(out, np.ndarray)

def test_freq_shift():
    sig = np.ones(100, dtype=complex)
    shift = FrequencyShift(1000, 48000)
    out = shift(sig)
    assert isinstance(out, np.ndarray)

def test_audio_normalize():
    sig = np.array([0.0, 1.0, -0.5])
    norm = AudioNormalize()
    out = norm(sig.copy())
    assert np.max(np.abs(out)) == pytest.approx(1.0)

def test_envelope_detector():
    data = np.random.randn(100)
    detector = EnvelopeDetector()
    envelope = detector(data)
    assert len(envelope) == len(data)
    assert np.all(envelope >= 0)

def test_noise_suppressor():
    data = np.random.randn(100)
    suppressor = NoiseSuppressor()
    filtered = suppressor(data)
    assert len(filtered) == len(data)

def test_decision_maker():
    decision_maker = DecisionMaker()
    envelope = np.ones(100) * 0.6
    snr = 15
    noise_free = True
    assert decision_maker(envelope, snr) is True
    snr = 5
    assert decision_maker(envelope, snr) is False
