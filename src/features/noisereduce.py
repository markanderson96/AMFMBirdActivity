import numpy as np
from pyfilterbank.octbank import FractionalOctaveFilterbank

class noiseReduction(object):
    def __init__(self, samplerate, window_size, window_overlap,
                 nth_oct, norm_freq, start_band, end_band, r_filters):
        self.samplerate = samplerate
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.num_filters = -(start_band) + end_band + 1
        self.r_filters = r_filters
        self.window = np.hanning(self.window_size)
        self.ofb = FractionalOctaveFilterbank(samplerate,
                                              order=8,
                                              nth_oct=nth_oct,
                                              norm_freq=norm_freq,
                                              start_band=start_band,
                                              end_band=end_band,
                                              filterfun='py')

        centers = self.ofb.center_frequencies
        centers = np.append(np.array([0]), centers)
        self.bandwidths = np.diff(centers * 2)

    def noiseReduction(self, data):
        """ time/frequency banks """
        window_size = int(self.samplerate * self.window_size)
        window_overlap = int(self.samplerate * self.window_overlap)

        r_data = np.zeros(len(data), dtype=np.float32)
        ii = 0

        while ii + window_size <= len(data):
            # create filterbank and filter data
            banks, _ = self.ofb.filter(data[ii:ii + window_size] *
                                            np.hanning(window_size))

            # for each band, estimate energy
            energies = np.array([])
            for j in range(self.num_filters):
                energy = np.inner(banks[:, j], 
                                  banks[:,j]) / self.bandwidths[j]
                energies = np.append(energies, energy)

            # reconstruct
            band_idx = np.argpartition(energies, -self.r_filters)[-self.r_filters:]
            active_bands = np.array([
                banks[:, k] if k in band_idx else np.zeros(window_size)
                for k in range(self.num_filters)
            ])
            r_data[ii:ii + window_size] += active_bands.sum(axis=0)
            ii += window_overlap

        return r_data