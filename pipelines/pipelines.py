import logging
import numpy as np
from scipy import *
from dsptools.dsptools import *


logging.basicConfig(level=logging.INFO)

class PipelineBuilder:
    def __init__(self):
        self.steps = []

    def add_step(self, func):
        self.steps.append(func)
        return self

    def build(self):
        def pipeline(data):
            for step in self.steps:
                data = step(data)
            return data
        return pipeline

class DSPipelines:
    def __init__(self, builder=PipelineBuilder):
        self.builder = builder

    def construct_noaa_apt_decoder(self, source, decoder, freq_shift=0, sourcetype='wav'):
        AUDIO_SAMPLE_RATE = 48000
        RESAMPLED_AUDIO_SAMPLE_RATE = 20800

        builder = self.builder()
        if sourcetype == 'hackrf': 
            builder.add_step(BytesToComplex(data_type=np.int16, normalize=True))
        elif sourcetype =='wav':
            builder.add_step(WavToComplex(normalize=True))
        else:
            raise ValueError("Unknown source type")
        builder.add_step(FrequencyShift(freq_shift, source.sample_rate))
        builder.add_step(LowPassFilterIIR(20000, source.sample_rate))
        builder.add_step(FMQuadratureDemod())
        builder.add_step(AutoDecimator(source.sample_rate, AUDIO_SAMPLE_RATE))
        builder.add_step(BandPassFilterFIR(10, 5000, AUDIO_SAMPLE_RATE))
        builder.add_step(AutoResampler(AUDIO_SAMPLE_RATE, RESAMPLED_AUDIO_SAMPLE_RATE))
        builder.add_step(decoder)
        return builder.build()
    
    #TODO
    def construct_noaa_signal_detector(self, carrier_freq, sat_frequencies, 
                                       tolerance=100000, snr_threshold=10, sourcetype='wav'):
        builder = self.builder()
        if sourcetype == 'hackrf': 
            builder.add_step(BytesToComplex(data_type=np.int16, normalize=True))
        elif sourcetype =='wav':
            builder.add_step(WavToComplex(normalize=True))
        else:
            raise ValueError("Unknown source type")
        
        self.carrier_freq = carrier_freq
        self.sat_frequencies = sat_frequencies
        self.tolerance = tolerance
        self.snr_threshold = snr_threshold
        # builder.add_step()
        # builder.add_step(EnvelopeDetector())
        # builder.add_step(dBmSignalPower())
        # builder.add_step(SignalNoiseRatio())
        # builder.add_step(DecisionMaker())