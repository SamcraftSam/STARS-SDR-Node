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

    def construct_noaa_apt_decoder(self, source, decoder, sourcetype='wav'):
        AUDIO_SAMPLE_RATE = 48000
        RESAMPLED_AUDIO_SAMPLE_RATE = 20800

        builder = self.builder()
        if sourcetype == 'hackrf': 
            builder.add_step(BytesToComplex(data_type=np.int16, normalize=True))
        elif sourcetype =='wav':
            builder.add_step(WavToComplex(normalize=True))
        else:
            raise ValueError("Unknown source type")
        builder.add_step(FrequencyShift(-180000, source.sample_rate))
        builder.add_step(LowPassFilterIIR(20000, source.sample_rate))
        builder.add_step(FMQuadratureDemod())
        builder.add_step(AutoDecimator(source.sample_rate, AUDIO_SAMPLE_RATE))
        builder.add_step(BandPassFilterFIR(10, 5000, AUDIO_SAMPLE_RATE))
        builder.add_step(AutoResampler(AUDIO_SAMPLE_RATE, RESAMPLED_AUDIO_SAMPLE_RATE))
        builder.add_step(decoder)
        return builder.build()
