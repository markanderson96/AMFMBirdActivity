import numpy as np
import math
import sys

from scipy.fft import fft, fftfreq, ifft
from scipy.signal import argrelmax, hilbert, resample, firwin, filtfilt
from scipy.stats import skew, kurtosis

class AM(object):
    """ 
    AM Calculation and related functions

    Attributes
    ----------
    data : np.array
        The data to be analysed
    samplerate : int
        Samplerate of data
    min_mod : float
        Minimum AM modulation frequency
    max_mod : float
        Maximum AM modulation frequency
    N : int
        Frequency bins in AM Power Spectrum
    envelope_Fs : int
        AM envelope samplerate
    prominence_cutoff: float
        Threshold parameter to determine whether AM is detected or not

    Methods
    -------
    getEnv(self)
        Extracts amplitude envelope via hilbert transform and resamples it
        to envelope_Fs
    calcAM(self)
        Calculate whether AM is detected, and if it is, it's frequency,
        modulation depth and prominence
    
    """
    def __init__(self, data, samplerate, min_mod, max_mod, prominence_cutoff):
        """ 
        Constructor for AM Class 

        Parameters
        ----------
        data : np.array
            The data to be analysed
        samplerate : int
            Samplerate of data
        min_mod : float
            Minimum AM modulation frequency
        max_mod : float
            Maximum AM modulation frequency
        prominence_cutoff: float
            Threshold parameter to determine whether AM is detected or not
        """

        self.data = data
        self.samplerate = samplerate
        self.min_mod = min_mod
        self.max_mod = max_mod
        self.N = int(3 * max_mod)
        self.envelope_Fs = self.N * 2
        self.prominence_cutoff = prominence_cutoff
        self.detected = []
        self.frequency = []
        self.depth = []
        self.prominence = []

    def getEnv(self):
        """ 
        Extracts the Envelope of a signal via Hilbert Transform
            
        Parameters
        ----------
        None
        
        Returns
        -------
        envelope : np.array
            Resampled amplitude envelope of data
        """

        envelope = np.abs(hilbert(self.data))
        newNumSamples = int(self.envelope_Fs *
                            (len(envelope) / int(self.samplerate)))
        envelope = resample(envelope, newNumSamples)
        b = firwin(16, [self.min_mod, self.max_mod],
                   pass_zero=False,
                   fs=self.envelope_Fs)
        envelope = filtfilt(b, 1.0, envelope)
        dcValue = np.mean(envelope)
        envelope -= dcValue

        return envelope

    def calcAM(self):
        """ 
        Determine in AM present, if so calculate AM Frequency, 
        AM Modulation Depth and AM Prominence.

        Parameters
        ----------
        None

        Returns
        -------
        am_detected : np.array(dtype=np.bool_)
            Array of booleans denoting whether AM is detected in 
            each block
        am_frequency : np.array(dtype=np.float64)
            If AM detected, contains frequency, otherwise 0
        am_mod_depth : np.array(dtype=np.float64)
            If AM detected, contains mod depth, otherwise 0
        am_prominence : np.array(dtype=np.float64)
            If AM detected, contains prominence, otherwise 0
        """

        envelope = self.getEnv()

        ii = 0
        while ii + self.envelope_Fs <= len(envelope):
            # detrend
            t = np.linspace(0, 1, self.envelope_Fs, endpoint=False)
            poly_coeff = np.polyfit(t, envelope[ii:ii + self.envelope_Fs], 3)  # coefficients for 3rd order fit
            envelope[ii:ii + self.envelope_Fs] -= np.polyval(poly_coeff, t)  # calculate and subtract

            # FFT
            i_freq = np.arange(1, self.N / 2, dtype=int)  # indices for frequency components
            freqs = fftfreq(self.N, 1.0 / self.envelope_Fs)  # frequencies to match output from DFT
            fft_out = fft(envelope[ii:ii + self.envelope_Fs])

            # calculate power spectrum
            ps = (abs(fft_out[i_freq])**2 + abs(fft_out[-i_freq])**2) / self.N/2**2
            freqs_ps = freqs[i_freq]

            # find i_max_ps
            i_max_ps = argrelmax(ps)[0]
            freqs_max = freqs_ps[i_max_ps]

            # indices of specified mod i_pos_freqs
            i_freq_in_range = (freqs_max >= self.min_mod) & (freqs_max <=
                                                        self.max_mod)
            freqsValid = freqs_max[i_freq_in_range]
            # if no peaks return nothing
            if not np.any(i_freq_in_range):
                self.detected.append(0)
                self.frequency.append(0)
                self.depth.append(0)
                self.prominence.append(0)
                ii += int(self.envelope_Fs / 2)
                continue

            # indices of valid peaks
            iPeaks = [np.where(freqs_ps == x)[0][0] for x in freqsValid]
            maxVal = max(ps[iPeaks])  # find highest peaks
            i_max_peak = np.where(ps == maxVal)[0][0]
            fundamental_freq = freqs_ps[
                i_max_peak]  # its the fundamental frequency

            # find peak prominence
            i_averages = [
                i_max_peak + x for x in [-3, -2, 2, 3]
                if i_max_peak + x in range(len(ps))
            ]
            average = np.average(ps[i_averages]) # average of frequencies around peak
            prominence = maxVal / float(average) # ratio of peak to average_around

            # check if prominence greater than threshold
            if prominence < self.prominence_cutoff:
                self.detected.append(0)
                self.frequency.append(0)
                self.depth.append(0)
                self.prominence.append(0)
                ii += int(self.envelope_Fs / 2)
                continue

            # inverse transform, need indices around peaks
            i_includes = [
                i_max_peak + x for x in [-1, 0, 1]
                if i_max_peak + x in range(len(ps))
            ]

            # inverse transform of fundamental
            i_pos_freqs =  i_freq[i_includes] 
            i_neg_freqs = -i_freq[i_includes] 
            fft_out_fundamental = np.zeros_like(fft_out) 
            fft_out_fundamental[i_pos_freqs] = fft_out[i_pos_freqs]
            fft_out_fundamental[i_neg_freqs] = fft_out[i_neg_freqs]
            fundamental_sine = np.real(ifft(fft_out_fundamental)) # inverse transform for fundamental

            # Check peak to peak value of fundamental sine wave should be greater than 1.5 dB
            fundamental_pp = max(fundamental_sine) - min(fundamental_sine)
            if (fundamental_pp > 1.5):
                for i in [2, 3]:
                    # estimate harmonics
                    harmonic_freq = fundamental_freq * i
                    i_harmonic = (np.abs(freqs_ps - harmonic_freq)).argmin()
                    harmonic_freq = freqs_ps[i_harmonic]

                    # check if local max
                    if i_harmonic not in i_max_ps:
                        # check surrounding indices
                        i_harmonic_new = None
                        if i == 2:
                            i_search = [-1, 0, 1]
                        else:
                            i_search = [-2, -1, 0, 1, 2]

                        # get indices from search
                        i_harmonic_search = [(i_harmonic + x) for x in i_search
                                           if (i_harmonic + x) in range(len(ps))
                                        ]

                        if np.any([x in i_max_ps for x in i_harmonic_search]):
                            i_surrounding_peaks = [
                                x for x in i_harmonic_search if x in i_max_ps
                            ]
                            vals_peaks = ps[i_surrounding_peaks]
                            i_harmonic_new = i_surrounding_peaks[
                                vals_peaks.tolist().index(max(vals_peaks))]

                        if i_harmonic_new:
                            i_harmonic = [
                                i_harmonic_new + x for x in i_search
                                if i_harmonic_new + x in range(len(ps))
                            ]
                        else:
                            continue
                    else:
                        # This means the estimated harmonic frequency is a local maximum
                        # Get indices around harmonic for inclusion in inverse transform
                        i_harmonic = [
                            i_harmonic + x for x in i_search
                            if i_harmonic + x in range(len(ps))
                        ]

                    # create harmonics if any
                    harmonic_sine = np.zeros_like(fft_out)
                    i_pos_freqs = i_freq[i_harmonic]
                    i_neg_freqs = -i_freq[i_harmonic]
                    harmonic_sine[i_pos_freqs] = fft_out[i_pos_freqs] 
                    harmonic_sine[i_neg_freqs] = fft_out[i_neg_freqs]  
                    harmonic_sine = np.real(ifft(harmonic_sine))
                    harmonic_pp = max(harmonic_sine) - min(harmonic_sine)

                    if (harmonic_pp > 1.5):
                        i_includes += i_harmonic

            # Create array with just fundamental and relevant harmonics
            fft_output_harmonics = np.zeros_like(fft_out)
            i_pos_freqs = i_freq[i_includes]
            i_neg_freqs = -i_freq[i_includes]
            fft_output_harmonics[i_pos_freqs] = fft_out[i_pos_freqs]  # first put the positive frequencies in
            fft_output_harmonics[i_neg_freqs] = fft_out[i_neg_freqs]  # next put the negative frequencies in

            # Inverse Fourier transform
            ifft_out = np.real(ifft(fft_output_harmonics))

            # Calculate mod depth using percentiles
            L5 = np.percentile(ifft_out, 95)
            L95 = np.percentile(ifft_out, 5)
            mod_depth = L5 - L95

            self.detected.append(1)
            self.frequency.append(fundamental_freq)
            self.depth.append(mod_depth)
            self.prominence.append(prominence)
            ii += int(self.envelope_Fs / 2)

        return self.detected, self.frequency, self.depth, self.prominence