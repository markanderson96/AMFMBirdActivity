import numpy as np

class parametric(object):
    def __init__(self, data, samplerate, window_length, window_overlap):
        self.data = data
        self.samplerate = samplerate
        self.window_length = int(window_length * self.samplerate)
        self.window_overlap = int(window_overlap * self.samplerate)

    def parametricFeatures(self):
        mSC = []
        vSC = []
        mR = []
        vR = []
        ii = 0
        while ii + self.samplerate <= len(self.data):
            block = self.data[ii:ii + self.samplerate] # get data block
            centroids = []
            rolloff = []
            jj = 0         
            while jj + self.window_length <= len(block):
                x = block[jj:jj + self.window_length]
                X = np.abs(np.fft.rfft(x))
                centroids.append(self.spectralCentroid(X))
                rolloff.append(self.rollOff(X))
                jj += int(self.window_overlap)

            mSC.append(np.mean(np.ma.masked_array(centroids, np.isnan(centroids))))
            vSC.append(np.var(np.ma.masked_array(centroids, np.isnan(centroids))))
            mR.append(np.mean(np.ma.masked_array(rolloff, np.isnan(rolloff))))
            vR.append(np.var(np.ma.masked_array(rolloff, np.isnan(rolloff))))

            del centroids
            del rolloff

            ii += int(self.samplerate/2)

        return mSC, vSC, mR, vR

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