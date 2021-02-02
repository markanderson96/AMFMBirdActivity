import numpy as np
import math
import sys
from dataclasses import dataclass, field

from scipy.fft import fft, fftfreq, ifft
from scipy.signal import argrelmax, hilbert, resample, firwin, filtfilt
from scipy.stats import skew, kurtosis

from pyfilterbank.octbank import FractionalOctaveFilterbank

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
    getHilbertEnv(self)
        Extracts amplitude envelope via hilbert transform and resamples it
        to envelope_Fs
    calcAM(self)
        Calculate whether AM is detected, and if it is, it's frequency,
        modulation depth and prominence
    _calcAMBand(self, data)
        Calculate wheter AM is present in a given band (internal method of calcAM() )
    
    """
    def __init__(self, data, samplerate, min_mod, max_mod, prominence_cutoff, depth_threshold):
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
        self.blocks = int((np.round(2 * len(self.data) / self.samplerate)) - 1)
        self.N = 100
        self.envelope_Fs = 100
        self.prominence_cutoff = prominence_cutoff
        self.depth_threshold = depth_threshold

    def getHilbertEnv(self):
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

        return envelope
    
    def calcAM(self):
        """ 
        Determine in AM present in files/windows,
        if so calculate AM Frequency, 
        AM Modulation Depth and AM Prominence.


        Goes through several frequency bands to determine best
        envelope to extract

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
        _freqs = [500,630,800,1000,1250,1600,2000,2500,3150,4000,5000,6300,8000,10000,12500,16000]
        freqs_i = [-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12]

        @dataclass
        class results:
            detected: list[int] = field(default_factory=list)
            frequency: list[float] = field(default_factory=list)
            depth: list[float] = field(default_factory=list)
            prominence: list[float] = field(default_factory=list)
            env: list[float] = field(default_factory=list)
        
        result = results()

        band_1 = results()
        band_2 = results()
        band_3 = results()
        band_4 = results()

        results_list = [band_1, band_2, band_3, band_4]

        for i in range(4):
            k1 = freqs_i.index(i * 3 - 3)
            k2 = freqs_i.index(i * 3 + 3)
            ofb = FractionalOctaveFilterbank(sample_rate=self.samplerate, 
                                            order=8, nth_oct=3.0, norm_freq=1000,
                                            start_band=freqs_i[k1],
                                            end_band=freqs_i[k2], filterfun='py')

            Leq = []
            samples_per_10ms = round(self.samplerate/100)
            slices_in_file = math.floor(len(self.data) / samples_per_10ms)
            
            for x in range(slices_in_file):
                idx = x * samples_per_10ms
                bands, _states = ofb.filter(self.data[idx:idx + samples_per_10ms])
                L = np.sqrt(np.mean(bands*bands))
                Leq.append(L)

            results_list[i].env = Leq

            ii = 0
            while ii + 100 <= len(Leq):
                results = self._calcAMBand(Leq[ii:ii+100])
                results_list[i].detected += [results['detected']]
                results_list[i].frequency += [results['freq']]
                results_list[i].depth += [results['depth']]
                results_list[i].prominence += [results['prominence']]
                ii += 50

            del Leq

        for i in range(self.blocks):
            band_of_interest = 0
            max_prominence = 0
            for j in range(4):
                prominence = np.max(results_list[j].prominence[i])
                if prominence > max_prominence:
                    band_of_interest = j
                    max_prominence = prominence
        
            result.detected.append(results_list[band_of_interest].detected[i])
            result.frequency.append(results_list[band_of_interest].frequency[i])
            result.depth.append(results_list[band_of_interest].depth[i])
            result.prominence.append(results_list[band_of_interest].prominence[i])
            result.env.extend(results_list[band_of_interest].env[i * 50 : i * 50 + 100])
        
        return result.detected, result.frequency, result.depth, result.prominence, result.env

    def _calcAMBand(self, Leq):
        """ 
        Determine in AM present, if so calculate AM Frequency, 
        AM Modulation Depth and AM Prominence.

        Parameters
        ----------
        Leq : np.array(dtype=np.float64)
            Time series of envelope to calculate AM in

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

        # some constants
        block_in_secs = 1
        fs = self.envelope_Fs
        N = self.N

        # detrend
        t = np.linspace(0, block_in_secs, len(Leq), endpoint=False)
        poly_coeff = np.polyfit(t, Leq, 3)  # coefficients for 3rd order fit
        Leq -= np.polyval(poly_coeff, t)  # calculate and subtract

        # FFT
        i_freq = np.arange(1, N / 2, dtype=int)  # indices for frequency components
        freqs = fftfreq(N, 1.0 / fs)  # frequencies to match output from DFT
        fft_out = fft(Leq)

        # calculate power spectrum
        ps = (abs(fft_out[i_freq])**2 + abs(fft_out[-i_freq])**2) / self.N/2**2
        freqs_ps = freqs[i_freq]

        # find i_max_ps
        i_max_ps = argrelmax(ps)[0]
        freqs_max = freqs_ps[i_max_ps]

        # indices of specified mod i_pos_freqs
        i_freq_in_range = (freqs_max >= self.min_mod) & (freqs_max <= self.max_mod)
        freqsValid = freqs_max[i_freq_in_range]
        # if no peaks return nothing
        if not np.any(i_freq_in_range):
            results = {'detected'    : 0,
                       'freq'        : 0,
                       'depth'       : 0,
                       'prominence'  : 0
            }
            return results

        # indices of valid peaks
        iPeaks = [np.where(freqs_ps == x)[0][0] for x in freqsValid]
        maxVal = np.max(ps[iPeaks])  # find highest peaks
        i_max_peak = np.where(ps == maxVal)[0][0]
        fundamental_freq = freqs_ps[i_max_peak]  # its the fundamental frequency

        # find peak prominence
        i_averages = [
            i_max_peak + x for x in [-3, -2, 2, 3]
            if i_max_peak + x in range(len(ps))
        ]
        average = np.average(ps[i_averages]) # average of frequencies around peak
        prominence = maxVal / float(average) # ratio of peak to average_around

        # check if prominence greater than threshold
        if prominence < self.prominence_cutoff:
            results = {'detected'    : 0,
                       'freq'        : 0,
                       'depth'       : 0,
                       'prominence'  : 0
            }
            return results

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

        i_search = [-1, 0, 1]
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

        if mod_depth < self.depth_threshold:
            results = {'detected'    : 0,
                       'freq'        : 0,
                       'depth'       : 0,
                       'prominence'  : 0
            }
        else:
            results = {'detected'    : 1,
                       'freq'        : fundamental_freq,
                       'depth'       : mod_depth,
                       'prominence'  : prominence
            }

        return results