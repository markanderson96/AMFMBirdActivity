import numpy as np

class AD():
    """ Rudimentary VAD, used as a first pass in the detections of bird activity """
    def __init__(self,
                 data,
                 samplerate,
                 window_length,
                 window_overlap,
                 block_size,
                 threshold,
                 band_start,
                 band_end):
        """ Constructor for VAD Class """

        self.data = data
        self.samplerate = samplerate
        self.window_length = window_length  # 20ms
        self.window_overlap = window_overlap  # Overlap 50%
        self.block_size = block_size  # length of block_size to detect speech in
        self.threshold = threshold  # TODO change to adaptive VAD this is for testing a simple version
        self.bird_band_start = band_start
        self.bird_band_end = band_end

    def calculateEnergy(self, data):
        """ Calculate Energy in signal chunk
        
            Args:
                * self - use member class variables
                * data - data to be processed
            Returns:
                * energy - energy of signal """

        data_amplitude = np.abs(np.fft.fft(data))
        data_amplitude = data_amplitude[1:]
        data_energy = data_amplitude**2
        return data_energy

    def calculateNormalisedEnergy(self, data):
        """ Calculate Normalised Energy in signal chunk
        
            Args:
                * self - use member class variables
                * data - data to be processed
            Returns:
                * energy - energy of signal normalised for frequency range """

        data_freq = np.fft.fftfreq(len(data), 1.0 / self.samplerate)
        data_freq = data_freq[1:]
        data_energy = self.calculateEnergy(data)

        energy_freq = {}
        for (i, freq) in enumerate(data_freq):
            if abs(freq) not in energy_freq:
                energy_freq[abs(freq)] = data_energy[i] * 2

        return energy_freq

    def sumEnergyBand(self, energyFreq):
        """ Sum Energy in relevant frequency bands
        
            Args:
                * self - use member class variables
                * energyFreq - energy data to be processed
            Returns:
                * sumEnergy - summed energy of signal in bands of interest """

        sumEnergy = 0
        for f in energyFreq.keys():
            if self.bird_band_start < f < self.bird_band_end:
                sumEnergy += energyFreq[f]
        return sumEnergy

    def smoothDetection(self, detectedWindows):
        """ Median Filter to smooth out activations
        
            Args:
                * self - use member class variables
                * datdetectedWindows - data to be processed
            Returns:
                * Median Filtered activations """

        median_window = int(self.block_size / self.window_length)
        if median_window % 2 == 0:
            median_window = median_window - 1

        x = detectedWindows[:, 1]
        k = median_window

        assert k % 2 == 1, "Median filter length must be odd"
        assert x.ndim == 1, "Input must be one dimensional"
        k2 = (k - 1) // 2
        y = np.zeros((len(x), k), dtype=x.dtype)
        y[:, k2] = x
        for i in range(k2):
            j = k2 - i
            y[j:, i] = x[:-j]
            y[:j, i] = x[0]
            y[:-j, -(i + 1)] = x[j:]
            y[-j:, -(i + 1)] = x[-1]

        return np.median(y, axis=1)

    def detectActivity(self):
        """ Detects possible areas of activity based on a ratio between specified band energy
            and whole signal using a static threshold. 
            
            Args:
                * self - use member class variables
            Returns:
                * Windows of Detected Activity """

        detected_windows = np.array([])
        sample_window = int(self.samplerate * self.window_length)
        sample_overlap = int(self.samplerate * self.window_overlap)
        data = self.data
        sample_start = 0

        while (sample_start < (len(data) - sample_window)):
            sample_end = sample_start + sample_window
            if sample_end >= len(data):
                sample_end = len(data) - 1

            data_window = data[sample_start:sample_end] * np.hamming(
                sample_end - sample_start)
            energy_freq = self.calculateNormalisedEnergy(data_window)
            sum_bird_energy = self.sumEnergyBand(energy_freq)
            sum_energy = sum(energy_freq.values())

            bird_ratio = sum_bird_energy / sum_energy
            #print(bird_ratio)
            detected = bird_ratio > self.threshold
            detected_windows = np.append(detected_windows,
                                         [sample_start, detected])

            sample_start += sample_overlap

        detected_windows = detected_windows.reshape(
            int(len(detected_windows) / 2), 2)
        detected_windows[:, 1] = self.smoothDetection(detected_windows)
        return detected_windows

    def reconstruct(self):
        """ Reconstructs signal using VAD windows
            
            Args:
                * self - use member class variables
            Returns:
                * reconstructed_data - reconstructed signal """

        data = self.data
        detected_windows = self.detectActivity()
        reconstructed_data = np.zeros(len(data))
        reconstructed_window = np.zeros(len(data))
        sample_overlap = int(self.samplerate * self.window_overlap)
        window_length = int(self.samplerate * self.window_length)
        window = np.hamming(window_length)

        for i in range(len(detected_windows[:, 1])):
            sample = i * sample_overlap
            detected = detected_windows[i, 1]
            reconstructed_window[sample:sample + window_length] += window * detected

        reconstructed_data = np.multiply(data, reconstructed_window)
        return reconstructed_data