import numpy as np
import math
import sys

from scipy.fft import fft, fftfreq, ifft
from scipy.signal import argrelmax, hilbert, resample, firwin, filtfilt
from scipy.stats import skew, kurtosis

class FM(object):
    """ 
    FM Calculation and related functions 

    Attributes
    ----------
    data : np.array
        The data to be analysed
    samplerate : int
        Samplerate of data
    winbdow_length : int
        Legth of Yin analysis window in samples
    window_overlap : int
        Yin analysis window overlap in samples
    
    Subclasses
    ----------
    _Yin
        Implementation of Yin algorithm for pitch detection, only used internally

    Methods
    -------
    calcFM(self)
        Calculate statistical moments relating to change in pitch over block
    """
    def __init__(self, data, samplerate, 
                 window_length, window_overlap, 
                 threshold):
        """ 
        Constructor for FM Class 
        
        Parameters
        ----------
        data : np.array
            The data to be analysed
        samplerate : int
            Samplerate of data
        window_length : float
            Minimum AM modulation frequency
        window_overlap : float
            Maximum AM modulation frequency
        threshold : float
            Threshold parameter for Yin
        fps : int
            Frames per second (i.e. frames per label)
        """
        self.data = data
        self.samplerate = samplerate
        self.window_length = int(window_length * samplerate)
        self.window_overlap = int(window_overlap * samplerate)
        self.yin = self._Yin(self.window_length, self.samplerate, threshold)
        self.fps = int(1/window_length)

    def calcFM(self):
        """ 
        Make use of YIN Pitch tracking algorithm to create a set of pitch data.
        To reconcile window lengths and extract a more meaningful representation
        of frequency modulation statistics are used, namely mean, variance, skew
        and kurtosis. 

        Parameters
        ----------
        None

        Returns
        -------
        pitch_means : np.array(dtype=np.float64)
            Array of pitch estimate means, first moment
        pitch_var : np.array(dtype=np.float64)
            Array of pitch estimate variances, second moment
        pitch_skew : np.array(dtype=np.float64)
            Array of pitch estimate skewness, third moment
        pitch_kurtosis : np.array(dtype=np.float64)
            Array of pitch estimate kurtosis, fourth moment
        """

        pitches = []
        pitch_means = []
        pitch_var = []
        pitch_skew = []
        pitch_kurtosis = []

        data = self.data
        window_length = self.window_length
        window_overlap = self.window_overlap
        window = np.hamming(window_length)
        fps = self.fps

        ii = 0
        while ii + window_length <= len(data):
            windowed_data = data[ii:ii + window_length] * window
            pitch_est = self.yin.pitchTracking(windowed_data)
            pitches.append(pitch_est)
            ii += window_overlap

        jj = 0
        while jj + fps <= len(pitches):
            pitch_means.append(
                np.mean(
                    np.ma.masked_array(pitches[jj:jj + fps],
                                       np.isnan(pitches[jj:jj + fps])))) 
            pitch_var.append(
                np.std(
                    np.ma.masked_array(
                        pitches[jj:jj + fps],
                        np.isnan(pitches[jj:jj + fps])))) 
            pitch_skew.append(skew(pitches[jj:jj + fps]))
            pitch_kurtosis.append(kurtosis((pitches[jj:jj + fps])))
            jj += fps

        return pitch_means, pitch_var, pitch_skew, pitch_kurtosis

    class _Yin(object):
        """ 
        Implementation of YIN in Python

        Attributes
        ----------
        frame_size : int
            Length of Yin analysis frame in samples
        samplerate : int
            Samplerate of data
        threshold : float
            Yin threshold parameter
        yin_buffer_size : int
            Length of buffer in samples (frame_size/2)

        Methods
        -------
        pitchTracking(data)
            Pitch tracking function
        """
        # constructors
        def __init__(self, frame_size, samplerate, threshold):
            """ 
            Constructor for YIN class 
            
            Parameters
            ----------
            frame_size : int
                Length of Yin analysis frame in samples
            samplerate : int
                Samplerate of data
            threshold : float
                Yin threshold parameter
            """

            self.frame_size = frame_size
            self.samplerate = samplerate
            self.threshold = threshold
            self.yin_buffer_size = int(frame_size / 2)

        # YIN component funtions
        # Difference function to determine periodicity
        # Based on eqn (7) in original YIN paper
        def _fastDifference(self, data):
            """ 
            Fast Difference Function Calculator for YIN.
            
            See eqn(7) in original YIN paper (2002).
            
            Makes use of complex multiplication and FFT to bring
            complexity down from O(N^2) to O(NlogN). 
            
            Parameters
            ----------
            data : np.array(dtype=np.float32)
                Data to get pitch estimates of

            Returns
            -------
            buffer : np.array(dtype=np.float32)
                Autocorrelation (i.e. difference)
            """

            yin_buffer_size = self.yin_buffer_size
            frame_size = 2 * yin_buffer_size

            # allocate and intialise
            buffer = np.zeros((yin_buffer_size, ), dtype=np.float64)
            terms = np.zeros((yin_buffer_size, ), dtype=np.float64)

            kernel = np.zeros((frame_size, ), dtype=np.float64)
            acf_real = np.zeros((frame_size, ), dtype=np.float64)
            acf_imag = np.zeros((frame_size, ), dtype=np.float64)

            # power terms calculations
            for i in range(yin_buffer_size):
                terms[0] += data[i] * data[i]
            for tau in range(1, yin_buffer_size):
                terms[tau] = terms[tau - 1] - data[tau - 1] * data[
                    tau - 1] + data[tau + yin_buffer_size] * data[tau + yin_buffer_size]

            # autocorrelation via FFT
            # data
            F = np.fft.fft(data, frame_size)
            F_real = F.real
            F_imag = F.imag

            # other half of data as convolution 'kernel'
            for j in range(yin_buffer_size):
                kernel[j] = data[yin_buffer_size - 1 - j]
            K = np.fft.fft(kernel, frame_size)
            KReal = K.real
            KImag = K.imag

            # convolution via FFT
            for k in range(frame_size):
                acf_real[k] = F_real[k] * KReal[k] - F_imag[k] * KImag[k]
                acf_imag[k] = F_real[k] * KImag[k] + F_imag[k] * KReal[k]
            acf = np.array(acf_real, dtype=np.float64) + np.array(acf_imag, dtype=np.float64) * 1j
            inverse = np.fft.ifft(acf, frame_size)

            for j in range(yin_buffer_size):
                buffer[j] = terms[0] + terms[j] - 2 * inverse.real[j + yin_buffer_size - 1]

            return buffer

        # Cumulative Mean normalised differences
        def _cumulativeDifference(self, yin_buffer):
            """ 
            Cumulative Difference, takes a normalised, mean difference
            of the autocorrelation funtions. 
                
            Parameters
            ----------
            yin_buffer : np.array(dtype=np.float32)
                Yin data buffer

            Returns
            -------
            yin_buffer : np.array(dtype=np.float32)
                Normalised autocorrelation (i.e. difference)
            """

            yin_buffer[0] = 1.0
            running_sum = 0

            for tau in range(1, self.yin_buffer_size):
                running_sum += yin_buffer[tau]
                if running_sum == 0:
                    yin_buffer[tau] = 1
                else:
                    yin_buffer[tau] *= tau / running_sum

            return yin_buffer

        def _absoluteThreshold(self, yin_buffer):
            """ 
            Absolute Threshold of the tau values, identify the period from this.
            
            Parameters
            ----------
            yin_buffer : np.array(dtype=np.float32)
                Yin data buffer, post normalisation

            Returns
            -------
            tau : float
                First estimate of period (in samples), if invalid set to negative value (-1)        
            """

            for tau in range(0, self.yin_buffer_size):
                if yin_buffer[tau] < self.threshold:
                    while ((tau + 1) < self.yin_buffer_size
                           and yin_buffer[tau + 1] < yin_buffer[tau]):
                        tau += 1
                    break

            if (tau == self.yin_buffer_size or yin_buffer[tau] >= self.threshold):
                tau = -1

            return tau

        def _parabolicInterpolation(self, tau, yin_buffer):
            """ 
            Parabolic interpretation to hone in on precise tau value.

            See original Yin Paper for details.
            
            Parameters
            ----------
            tau : float
                Estimate of period (in samples) 
            yin_buffer : np.array(dtype=np.float32)
                Yin data buffer, post normalisation

            Returns
            -------
            better_tau : float
                A better estimate for tau after parabolic interpolation
            """

            if tau == self.yin_buffer_size:  # this isn't valid
                return tau

            better_tau = 0.0
            if tau > 0 and tau < self.yin_buffer_size - 1:  #this is valid
                s0 = yin_buffer[tau - 1]
                s1 = yin_buffer[tau]
                s2 = yin_buffer[tau + 1]

                adjustment = (s2 - s0) / (2 * (2 * s1 - s2 - s0))

                if np.fabs(adjustment) > 1: adjustment = 0

                better_tau = tau + adjustment
            else:
                print("WARNING: Can't interpolate at edge, will return uninterpolated value")
                better_tau = tau

            return better_tau

        def pitchTracking(self, data):
            """ 
            YIN pitch tracking algorithm, extracts F0 values from given data
            
            Parameters
            ----------
            data : np.array(dtype=np.float32)
                Data which is to have pitch estimation performed on it

            Returns
            -------
            pitch : float
                F0 at this window
            """

            # Calculate pitch
            # Step 1 - difference
            yin_buffer = self._fastDifference(data)
            # Step 2 - Cumulative Difference
            yin_buffer = self._cumulativeDifference(yin_buffer)
            # Step 3 - Absolute Thresholding
            tau_est = self._absoluteThreshold(yin_buffer)

            if (tau_est != -1):
                # Step 4 - Parabolic Interpolation
                better_tau = self._parabolicInterpolation(tau_est, yin_buffer)
                pitch = self.samplerate / better_tau
            else:
                pitch = 0

            return pitch