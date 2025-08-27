import logging
import numpy as np
from scipy import *
from dsptools import *

logging.basicConfig(level=logging.INFO)

class RadioPipelineBuilder:
    def __init__(self):
        self.steps = []

    def add_step(self, func):
        self.steps.append(func)
        return self

    def add_filter(self, func):      return self.add_step(func)
    def add_demodulator(self, func): return self.add_step(func)
    def add_decoder(self, func):     return self.add_step(func)
    def add_sink(self, func):        return self.add_step(func)

    def build(self):
        def pipeline(data):
            for step in self.steps:
                data = step(data)
            return data
        return pipeline

class DSPipelines:
    def __init__(self, builder):
        self.builder = builder

    #example
    def construct_fsk_pipeline(self, sink):
        sample_rate = 2_000_000
        cutoff = 100_000   
        return (self.builder
                .add_filter(LowPassFilter(cutoff, sample_rate))
                .add_demodulator(FMQuadratureDemod(gain=1.2))
                .add_decoder(PacketDecoder(threshold=0.4))
                .add_sink(sink)
                .build())
