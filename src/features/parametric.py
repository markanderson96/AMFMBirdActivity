import numpy as np

class parametric(object):
    def __init__(self, data, samplerate):
        self.data = data
        self.samplerate = samplerate

    def parametricFeatures(self):
        centroids = []
        rolloff = []
        ii = 0
        while ii + self.samplerate <= len(self.data):
            x = self.data[ii:ii + self.samplerate] # get data block
            X = np.abs(np.fft.rfft(x)) 
            centroids.append(self.spectralCentroid(X))
            rolloff.append(self.rollOff(X))
            ii += int(self.samplerate/2)
        return centroids, rolloff

    def spectralCentroid(self, X):
        length = len(X)
        freqs = np.abs(np.fft.fftfreq(length, 1.0/self.samplerate)) # positive frequencies
        centroid = np.sum(X*freqs) / np.sum(X) # weighted mean
        return centroid

    def rollOff(self, X):
        norm = X.sum() # get normalisation value
        if norm == 0:
            norm = 1
        #norm[norm == 0] = 1 # prevent division by 0
        X = np.cumsum(X) / norm # normalised cumulative sum
        vsr = np.argmax(X >= 0.95)
        vsr = vsr / (X.shape[0] - 1) * (self.samplerate / 2)
        return vsr