"""
MIT License

Copyright (c) 2018 Zac Stewart, Martin Bernardi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import numpy
import scipy.io.wavfile
import scipy.signal
import sys
import PIL
from PIL import Image


class APT(object):

    RATE = 20800
    NOAA_LINE_LENGTH = 2080

    def __init__(self, filename):
        (rate, self.signal) = scipy.io.wavfile.read(filename)
        if rate != self.RATE:
            raise Exception("Resample audio file to {}".format(self.RATE))

        # Keep only one channel if audio is stereo
        if self.signal.ndim > 1:
            self.signal = self.signal[:, 0]

        truncate = self.RATE * int(len(self.signal) // self.RATE)
        self.signal = self.signal[:truncate]

    """ 
        Modified, now it is possible to decode image without
        displaying the result
    """
    def decode(self, signal=None, outfile=None, imgshow=False):
        
        if signal is not None:
            self.signal = signal
            

        if self.signal is None or len(self.signal) == 0:
            raise ValueError("APT.decode(): signal is empty!")

        hilbert = scipy.signal.hilbert(self.signal)
        filtered = scipy.signal.medfilt(numpy.abs(hilbert), 5)
        reshaped = filtered.reshape(len(filtered) // 5, 5)
        digitized = self._digitize(reshaped[:, 2])
        matrix = self._reshape(digitized)

        if matrix.dtype != numpy.uint8:
            matrix = (numpy.clip(matrix, 0, 1) * 255).astype(numpy.uint8)
        image = Image.fromarray(matrix)
        if not outfile is None:
            image.save(outfile)
        if imgshow: 
            image.show() 
        return matrix

    def _digitize(self, signal, plow=0.5, phigh=99.5):
        '''
        Convert signal to numbers between 0 and 255.
        '''
        (low, high) = numpy.percentile(signal, (plow, phigh))
        delta = high - low
        data = numpy.round(255 * (signal - low) / delta)
        data[data < 0] = 0
        data[data > 255] = 255
        return data.astype(numpy.uint8)

    def _reshape(self, signal):
        '''
        Find sync frames and reshape the 1D signal into a 2D image.

        Finds the sync A frame by looking at the maximum values of the cross
        correlation between the signal and a hardcoded sync A frame.

        The expected distance between sync A frames is 2080 samples, but with
        small variations because of Doppler effect.
        '''
        # sync frame to find: seven impulses and some black pixels (some lines
        # have something like 8 black pixels and then white ones)
        syncA = [0, 128, 255, 128]*7 + [0]*7

        # list of maximum correlations found: (index, value)
        peaks = [(0, 0)]

        # minimum distance between peaks
        mindistance = 2000

        # need to shift the values down to get meaningful correlation values
        signalshifted = [x-128 for x in signal]
        syncA = [x-128 for x in syncA]
        for i in range(len(signal)-len(syncA)):
            corr = numpy.dot(syncA, signalshifted[i : i+len(syncA)])

            # if previous peak is too far, keep it and add this value to the
            # list as a new peak
            if i - peaks[-1][0] > mindistance:
                peaks.append((i, corr))

            # else if this value is bigger than the previous maximum, set this
            # one
            elif corr > peaks[-1][1]:
                peaks[-1] = (i, corr)

        # create image matrix starting each line on the peaks found
        matrix = []
        for i in range(len(peaks) - 1):
            matrix.append(signal[peaks[i][0] : peaks[i][0] + 2080])

        return numpy.array(matrix)


if __name__ == '__main__':
    apt = APT(sys.argv[1])

    if len(sys.argv) > 2:
        outfile = sys.argv[2]
    else:
        outfile = None
    apt.decode(outfile=outfile, imgshow=True)