import numpy as np
import scipy
import scipy.signal
import scipy.stats
from obspy import Stream, Trace
from obspy import read
import matplotlib.pyplot as plt
import SeisRoutine.plot as seisplot
import logging
import re
import os
import glob
import obspy as obs
import pywt
from dataclasses import dataclass
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view


class SlidingWindowProcessor:
    """A utility class to segment 1D arrays into sliding windows

    using static methods without needing class instantiation.
    """

    @staticmethod
    def _validate_inputs(data, window: int, step: int) -> np.ndarray:
        if window <= 0:
            raise ValueError("window must be a positive integer.")
        if step <= 0:
            raise ValueError("step must be a positive integer.")

        arr = np.asarray(data)
        if arr.ndim != 1:
            raise ValueError("Input data must be a 1D array.")
        if len(arr) < window:
            raise ValueError(
                f"Data length ({len(arr)}) cannot be smaller than window ({window})."
            )

        return arr

    @staticmethod
    def transform_iterative(data, window: int, step: int = 1):
        """Extract windows using standard iteration (list of arrays)."""
        arr = SlidingWindowProcessor._validate_inputs(data, window, step)

        windows = []
        start_indices = []

        for start in range(0, arr.size - window + 1, step):
            windows.append(arr[start : start + window])
            start_indices.append(start)

        return windows, start_indices

    @staticmethod
    def transform_vectorized(data, window: int, step: int = 1):
        """Extract windows using numpy.lib.stride_tricks.sliding_window_view."""
        arr = SlidingWindowProcessor._validate_inputs(data, window, step)

        all_windows = sliding_window_view(arr, window_shape=window)
        windows = all_windows[::step]

        num_windows = len(windows)
        start_indices = np.arange(0, num_windows * step, step)

        return windows, start_indices

    @staticmethod
    def transform(data, window: int, step: int = 1, method: str = "vectorized"):
        """Unified interface to extract windows using the chosen method."""
        if method == "vectorized":
            return SlidingWindowProcessor.transform_vectorized(
                data, window, step
            )
        elif method == "iterative":
            return SlidingWindowProcessor.transform_iterative(
                data, window, step
            )
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose 'vectorized' or 'iterative'."
            )


@dataclass
class SpikeDetectionResult:
    detected: bool
    spike_indices: np.ndarray
    spike_amplitudes: np.ndarray
    values: dict


class SpikeDetector:

    def __init__(
            self,
            data
        ):
        self.data = np.asarray(data, dtype=float).ravel()

    def _build_result(
            self,
            spike_indices,
            **values
        ):
        spike_indices = np.asarray(spike_indices, dtype=int)
        return SpikeDetectionResult(
            detected=len(spike_indices) > 0,
            spike_indices=spike_indices,
            spike_amplitudes=self.data[spike_indices]
            if len(spike_indices)
            else np.array([], dtype=self.data.dtype),
            values=values,
        )

    def zscore(
            self,
            threshold=10
        ):
        z_score = scipy.stats.zscore(self.data)
        spikes = np.where(z_score > threshold)[0]
        result = self._build_result(
            spikes,
            zscore=z_score,
            threshold=threshold,
        )
        return result

    def differential(
            self,
            dt=0.01,
            threshold=100.0
        ):
        diffs = np.abs(np.diff(self.data)) / dt
        spikes = np.where(diffs > threshold)[0]
        result = self._build_result(
            spikes,
            differential=diffs,
            threshold=threshold,
            dt=dt,
        )
        return result

    def mad(
            self,
            threshold=6
        ):
        normalizing_factor = 0.6745
        median = np.median(self.data)
        mad = np.median(np.abs(self.data - median))
        modified_zscore = np.abs(
            normalizing_factor *
            (self.data - median) /
            (mad + 1e-8)
        )
        spikes = np.where(modified_zscore > threshold)[0]
        result = self._build_result(
            spikes,
            median=median,
            mad=mad,
            modified_zscore=modified_zscore,
            threshold=threshold,
        )
        return result

    def prominence(
            self,
            prominence=5
        ):
        peaks, properties = scipy.signal.find_peaks(
            np.abs(self.data),
            prominence=prominence,
        )
        result = self._build_result(
            peaks,
            prominence=properties["prominences"],
            threshold=prominence,
        )
        return result

    def wavelet(
            self,
            wavelet="db4",
            level=4,
            coeffs_index=-1,
            threshold=3.5,
        ):
        coeffs = pywt.wavedec(self.data, wavelet, level=level)
        detail = coeffs[coeffs_index]
        std = np.std(detail)
        spike_locs = np.where(np.abs(detail) > threshold * std)[0]
        factor = len(self.data) / len(detail)
        indices = np.round(spike_locs * factor).astype(int)
        result = self._build_result(
            indices,
            detail_coefficients=detail,
            std=std,
            threshold=threshold,
        )
        return result

    def variance(
            self,
            start_idx_noise=0,
            end_idx_noise=-1,
            threshold=5,
        ):
        noise = self.data[start_idx_noise:end_idx_noise]
        variance = noise.var().item()
        max_amplitude = np.abs(noise).max()
        if max_amplitude > threshold * variance:
            spike_idx = np.array([np.argmax(np.abs(noise)) + start_idx_noise])
        else:
            spike_idx = np.array([], dtype=int)

        result = self._build_result(
            spike_idx,
            variance=variance,
            max_amplitude=max_amplitude,
            threshold=threshold,
        )
        return result

    def hampel(
            self,
            window_size=161,
            n_sigmas=3,
        ):
        data = self.data.copy()
        half_window = window_size // 2
        spike_mask = np.zeros(len(data), dtype=bool)
        filtered = data.copy()
        for i in range(len(data)):
            start = max(0, i - half_window)
            end = min(i + half_window + 1, len(data))
            window = data[start:end]
            median = np.median(window)
            mad = np.median(np.abs(window - median))
            if mad == 0:
                continue
            if np.abs(data[i] - median) > n_sigmas * 1.4826 * mad:
                spike_mask[i] = True
                filtered[i] = median
        spikes = np.where(spike_mask)[0]

        result = self._build_result(
            spikes,
            filtered=filtered,
            spike_mask=spike_mask,
            window_size=window_size,
            n_sigmas=n_sigmas,
        )
        return result

    def skewness(
            self,
            threshold=5,
            preprocessing=False,
        ):
        data = self.data.copy()
        if preprocessing:
            data -= data.mean()
            data[~np.isfinite(data)] = 0
        s = scipy.stats.skew(data, bias=False)
        spikes = (
            np.array([np.argmax(np.abs(data))])
            if abs(s) > threshold
            else np.array([], dtype=int)
        )
        result = self._build_result(
            spikes,
            skewness=s,
            threshold=threshold
        )
        return result

    def kurtosis(
            self,
            threshold=10,
            fisher=False,
            preprocessing=False,
        ):
        data = self.data.copy()
        if preprocessing:
            data -= data.mean()
            data[~np.isfinite(data)] = 0
        k = scipy.stats.kurtosis(data, fisher=fisher, bias=False)
        spikes = (
            np.array([np.argmax(np.abs(data))])
            if k > threshold
            else np.array([], dtype=int)
        )
        result = self._build_result(
            spikes,
            kurtosis=k,
            threshold=threshold,
        )
        return result

    def min_max_ratio(
            self,
            threshold=0.5,
        ):
        data = self.data
        mean = data.mean()
        min_ = abs(data.min() - mean)
        max_ = abs(data.max() - mean)
        min_, max_ = min(min_, max_),  max(min_, max_)
        # print(f"Invalid values: min_={min_}, max_={max_}")
        if max_ == 0:
            ratio = np.inf
        else:
            ratio = min_ / max_
        spikes = (
            np.array([np.argmax(np.abs(data))])
            if ratio < threshold
            else np.array([], dtype=int)
        )
        result = self._build_result(
            spikes,
            ratio=ratio,
        )

        return result



class SpikeDetector2:
    """
    A class for detecting spikes in signals using various statistical methods.
    All methods are defined as static.
    
    Example
    -----------
        # make signal
        from obspy import read
        import matplotlib.pyplot as plt
    
        st = read('http://examples.obspy.org/RJOB_061005_072159.ehz.new')
        tr = st[0]
        times = tr.times()
        signal = tr.data
        signal[16700] = 5e4
        sps = int(tr.stats.sampling_rate)
        
        # OR
        
        # signal = np.random.random(1000)
        # signal[792] = 100
        # sps = 200
    
        kwargs_sliding={
            "window": 4*sps,
            "step": 1*sps,
            "method": "vectorized",
        }
        kwargs_spike_suspected={
            "threshold": 2
        }
    
        all_peaks = srw.SpikeDetector2.detect(
            signal=signal,
            kwargs_sliding=kwargs_sliding,
            kwargs_spike_suspected=kwargs_spike_suspected,
        )
    
        plt.plot(times, signal)
        plt.scatter(x=times[all_peaks], y=signal[all_peaks], color='r')
        plt.show()
    """

    @staticmethod
    def is_spike_suspected_using_hampel(
        window,
        n_sigmas=3.0,
        check_any=False,
        center_idx=None
    ):
        """
        Detect spikes using Hampel filter.
        Based on:
            INSTANCE - the Italian seismic dataset for machine learning 
            https://doi.org/10.5194/essd-13-5509-2021
        This function was generated using Gemini.
        
        Check if a single window contains a spike using the Hampel/MAD
        criterion.

        Parameters:
            window (array-like):
                1D array representing a single window.
            n_sigmas (float):
                Detection threshold multiplier (default: 3.0).
            check_any (bool): 
                - False:
                    Only evaluates the target/center point (standard Hampel
                    approach).
                - True:
                    Checks if ANY point in the window exceeds the threshold.
            center_idx (int, optional):
                Index of the target point. Defaults to the window center.

        Returns:
            bool: True if a spike is detected, False otherwise.
        """
        w = np.asarray(window, dtype=float)
        if w.ndim != 1:
            raise ValueError("Input window must be a 1D array.")

        # 1. Compute window median and MAD
        median = np.median(w)
        mad = scipy.stats.median_abs_deviation(
            x=w,
            scale="normal",
        )

        # Flat line / zero variance check
        if mad == 0:
            return False

        threshold = n_sigmas * mad

        # Case A: Check if any element in the entire window is an outlier
        if check_any:
            has_spike = bool(np.any(np.abs(w - median) > threshold))
            return has_spike

        # Case B: Standard Hampel behavior (evaluate specific target point)
        if center_idx is None:
            center_idx = len(w) // 2

        target_diff = np.abs(w[center_idx] - median)
        is_spike = bool(target_diff > threshold)
        return is_spike

    @staticmethod
    def is_spike_suspected_using_skewness(
        window,
        threshold=2,
        bias=False,
    ):
        """
        Check if a window is suspected of containing a spike based on its
        skewness.
        
        Parameters:
            window (array-like):
                1D array representing a single window.
            threshold (float):
                Skewness threshold to flag a spike (default: 2).
            bias (bool):
                If False, then the calculations are corrected for statistical
                bias.
            
        Returns:
            bool:
                True if the absolute skewness exceeds the threshold,
                False otherwise.
        """
        skew = scipy.stats.skew(a=window, bias=bias)
        spike_suspicious = abs(skew) > threshold
        
        return spike_suspicious

    @staticmethod
    def detect(
        signal,
        kwargs_sliding={
            "window": None,
            "step": None,
            "method": "vectorized",
        },
        kwargs_spike_suspected={
            "threshold": 2
        },
    ):
        """
        Process the entire signal using sliding windows to detect spikes.
        
        Parameters:
            signal (array-like):
                The input 1D signal array.
            window_size (int):
                Size of the sliding window.
            step_size (int):
                Step size for moving the sliding window.
            
        Returns:
            list: A list of unique indices where spikes were detected.
        """
        windows, start_indices = SlidingWindowProcessor.transform(
            data=signal,
            **kwargs_sliding
        )

        all_spikes = []
        for start_index, window in zip(start_indices, windows):
            # Call the static method from within the class
            spike_suspicious = SpikeDetector2.is_spike_suspected_using_skewness(
                window=window,
                **kwargs_spike_suspected
            )
            
            if spike_suspicious:
                mad = scipy.stats.median_abs_deviation(x=window, scale=1.0)
                window_size = kwargs_sliding['window']
                peaks, properties = scipy.signal.find_peaks(
                    x=np.abs(window),
                    # x=window,
                    height=None,
                    threshold=None,
                    distance=window_size,
                    prominence=10*mad,
                    width=None,
                    wlen=None,
                    rel_height=0.5,
                    plateau_size=None
                )
                
                # Prevent ValueError when no peaks are found in the window
                if len(peaks) > 0:
                    all_spikes.append(peaks + start_index)
        
        # Concatenate and remove duplicate indices
        if all_spikes:
            all_spikes = np.concatenate(all_spikes)
            all_spikes = list(set(all_spikes))
        else:
            all_spikes = []
            
        return all_spikes


class SNR:
    """
    A utility class for computing different SNR estimators
    in time, frequency, statistical, and wavelet domains.
    """

    def __init__(
            self,
            data,
            sps,
            noise_window,
            signal_window,
        ):
        
        self.data = np.asarray(data)
        self.sps = sps
        self.noise_window = noise_window
        self.signal_window = signal_window

        # Ensure shape is (channels, samples)
        if self.data.ndim == 1:
            self.data = self.data[np.newaxis, :]


        self._extract_windows(
            noise_window,
            signal_window,
        )

    def _extract_windows(
            self,
            noise_window,
            signal_window,
        ):
        """
        Extract noise and signal segments from data.
        """

        n_samples = self.data.shape[1]

        sn, en = noise_window
        ss, es = signal_window

        sn = max(0, sn)
        ss = max(0, ss)

        en = min(en, n_samples)
        es = min(es, n_samples)

        if sn >= en:
            raise ValueError("Invalid noise_window.")
        if ss >= es:
            raise ValueError("Invalid signal_window.")
        self.noise = self.data[:, sn:en]
        self.signal = self.data[:, ss:es]
        
        self.noise = self.noise - self.noise.mean(axis=1, keepdims=True)
        self.signal = self.signal - self.signal.mean(axis=1, keepdims=True)
        
        # self.noise = self.noise.astype(np.float64)
        # self.signal = self.signal.astype(np.float64)
        # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        # ax1.plot(self.noise.T+[1, 0, -1]); ax1.set_title("Noise")
        # ax2.plot( self.signal.T+[1, 0, -1]); ax2.set_title("Signal")
        # plt.show()
        
    def peak_to_peak(
            self,
        ):
        max_noise = np.abs(self.noise).max()
        max_signal = np.abs(self.signal).max()
        
        return np.array([max_signal / max_noise])
    

    @staticmethod
    def _compute_power(
            data,
            axis=1,
            domain='time',
        ):
        
        n = data.shape[axis]

        if domain == 'time':
            # power = 1 / n * np.sum(np.abs(data) ** 2, axis=axis)
            power = np.mean(np.abs(data) ** 2, axis=axis)

        elif domain == 'frequency':
            power = 1 / (n ** 2) * np.sum(np.abs(data) ** 2, axis=axis)

        else:
            raise ValueError("domain must be 'time' or 'frequency'")
        
        return power

    def power_in_time(
            self,
            epsilon=1e-12,
            axis_power=1,
        ):

        p_signal = self._compute_power(
            data=self.signal,
            domain='time',
            axis=axis_power
        )
        p_noise = self._compute_power(
            data=self.noise,
            domain='time',
            axis=axis_power
        )
        p_noise = np.maximum(p_noise, epsilon)
        
        # print(
        #     f"noise: min: {self.noise.min()} max: {self.noise.max()}", self.noise.dtype, p_noise,
        #     f"signal: min: {self.signal.min()} max: {self.signal.max()}", self.signal.dtype, p_signal,
        #     "SNR", p_signal / p_noise
        # )


        return p_signal / p_noise

    def power_in_freq(
            self,
            epsilon=1e-12,
            axis_power=1,
        ):

        noise_fft = np.fft.fft(self.noise, axis=1)
        signal_fft = np.fft.fft(self.signal, axis=1)

        p_signal = self._compute_power(
            data=signal_fft,
            domain='frequency',
            axis=axis_power
        )
        p_noise = self._compute_power(
            data=noise_fft,
            domain='frequency',
            axis=axis_power
        )
        p_noise = np.maximum(p_noise, epsilon)

        return p_signal / p_noise

    def mad(
            self,
        ):

        noise_mad = scipy.stats.median_abs_deviation(self.noise, axis=1)
        signal_mad = scipy.stats.median_abs_deviation(self.signal, axis=1)

        return signal_mad / noise_mad

    def percentile(
            self,
            lbp=25,
            hbp=95,
            method=1,
        ):

        snr = []

        for signal, noise in zip(self.signal, self.noise):
            signal = np.abs(signal)
            noise = np.abs(noise)

            if method ==  1:

                signal_p = np.percentile(signal, hbp)
                noise_p = 1.4826 * scipy.stats.median_abs_deviation(noise)

            elif method == 2:

                signal = signal[
                    (signal >= np.percentile(signal, lbp)) &
                    (signal <= np.percentile(signal, hbp))
                ]
                signal_p = scipy.stats.median_abs_deviation(signal)

                noise = noise[
                    (noise >= np.percentile(noise, lbp)) &
                    (noise <= np.percentile(noise, hbp))
                ]
                noise_p = scipy.stats.median_abs_deviation(noise)

            snr.append(
                signal_p / noise_p
            )

        return np.asarray(snr)

    def cwt(
            self,
            scales=np.arange(1, 256),
            wavelet="morl",
        ):

        data = np.concatenate([self.noise, self.signal], axis=1)
        n_noise = self.noise.shape[1]

        snr = []

        for trace in data:
            coef, _ = pywt.cwt(
                trace,
                scales=scales,
                wavelet=wavelet,
                sampling_period=1 / self.sps,
            )

            energy = np.mean(np.abs(coef), axis=0)

            noise_energy = energy[:n_noise]
            signal_energy = energy[n_noise:]

            snr.append(
                np.mean(signal_energy) / np.mean(noise_energy)
            )

        return np.asarray(snr)


class StreamCache:
    def __init__(
            self,
            root: str=None,
            pattern_path: str=None,
            client_tsindex=None,
            client_sds=None,
            client_fdsn=None,
            merge_method=None,
            **pattern_vars
        ):
        self.root = root
        self.pattern_path = pattern_path
        self.client_tsindex = client_tsindex
        self.client_sds = client_sds
        self.client_fdsn = client_fdsn
        self.merge_method = merge_method
        self.pattern_vars = pattern_vars
        self.stream = None
        self.stations: list[str] = []
        self._loaded_julday: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(self, time, station_code: str):
        """Return the stream for a specific time and station.

        Reloads from disk only when the requested day differs from what
        is currently cached.
        """
        if self._should_reload(time):
            self._read(time)

        target_stream = self.stream.select(station=station_code)
        if not target_stream:
            msg = (
                f"Station '{station_code}' not found in stream for "
                f"year: {time.year} julday: {time.julday}."
            )
            logging.warning(msg)
            # raise ValueError(
            #     f"Station '{station_code}' not found in stream for "
            #     f"julday {time.julday}."
            # )
            return obs.Stream()
        return target_stream

    def get_by_pick(self, pick, after=0):
        """Convenience wrapper that extracts time and station from a Pick."""
        st = self.get(
            time=pick.time,
            station_code=pick.waveform_id.station_code
        )

        t = pick.time
        t_after = t+after
        if t.julday != (t_after).julday:
            pattern = self.pattern_path.format(
                time=t_after,
                **self.pattern_vars
            )
            pattern_path = f"{self.root}/{pattern}"
            st.extend(
                self._read_safely(pattern_path)
            )
            st.merge(-1)
            st.detrend("constant")
            if self.merge_method:
                st.merge(method=self.merge_method)

        return st
    
    def check_sps(self, sps=100):
        wrong = {
            tr.stats.station: tr.stats.sampling_rate
            for tr in self.stream
            if tr.stats.sampling_rate != sps
        }
        if wrong:
            raise ValueError(
                f"Unexpected sampling rate(s) — expected {sps} Hz, "
                f"got: {wrong}"
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _should_reload(self, time) -> bool:
        if self.stream is None:
            return True
        if time.julday != self._loaded_julday:
            return True
        return False

    def _read(self, time):
        if self.root and self.pattern:
            pattern = self.pattern_path.format(
                time=time,
                **self.pattern_vars,
            )
            pattern_path = Path(self.root) / pattern
            logging.info(f"Reading waveform data: {pattern_path}")

            self.stream = self._read_safely(pattern_path)

        elif self.client_fdsn:
            stime = obs.UTCDateTime(time)
            etime = stime + (24*60*60)
            msg = (
                "Loading data from FDSN client"
                f"Start Time: {stime}"
                f"End Time: {etime}"
            )
            logging.info(msg)
            self.stream = self.client_fdsn.get_waveforms(
                network="*",
                station="*",
                location="*",
                channel="*",
                starttime=stime,
                endtime=etime,
                merge=-1,
            )

        elif self.client_tsindex:
            stime = obs.UTCDateTime(time)
            etime = stime + (24*60*60)
            msg = (
                "Loading data from TSINDEX client"
                f"Start Time: {stime}"
                f"End Time: {etime}"
            )
            logging.info(msg)
            self.stream = self.client_tsindex.get_waveforms(
                network="*",
                station="*",
                location="*",
                channel="*",
                starttime=stime,
                endtime=etime,
                merge=-1,
            )

        elif self.client_sds:
            pass

        else:
            msg = "There isn't any"
            print(msg)
        self._preprocess()
        self._loaded_julday = self.stream[0].stats.starttime.julday
        self.stations = list({tr.stats.station for tr in self.stream})

    def _read_safely(self, path):
        '''
        There is an issue with some .gcf files that causes an error
        when attempting to read them with ObsPy.
        
        pattern: The pattern of the file. It must start with . and end with $.
        If you want the pattern to include "#", use "#".
        If you want the pattern to exclude "#", use "[^#]".
            Example: ".*6226z4.*20241201.*[^#].*\.gcf$"
        '''
        st = obs.Stream()
        lst_files = glob.glob(path)
        for fpath in lst_files:
            try:
                st.extend(obs.read(fpath))
                logging.info(f'Data Loaded | {fpath}')
            except Exception as error:
                logging.warning(f"Couldn't load data file | {fpath}")
                logging.debug(f"Becuase of the following error:\n{error}")    
        return st

    def _preprocess(self):
        self.stream.merge(-1)
        self.stream.detrend("constant")
        if self.merge_method:
            self.stream.merge(method=self.merge_method)




def read_gcf_safely(root: str, pattern: str):
    '''
    There is an issue with some .gcf files that causes an error
    when attempting to read them with ObsPy.
    
    pattern: The pattern of the file. It must start with . and end with $.
    If you want the pattern to include "#", use "#".
    If you want the pattern to exclude "#", use "[^#]".
        Example: ".*6226z4.*20241201.*[^#].*\.gcf$"
    '''
    st = Stream()
    pattern = re.compile(pattern)
    for path, dirs, files in os.walk(root):
        for fname in files:
            fpath = os.path.join(path, fname)
            if pattern.search(fpath):
                logging.info(f'Loading {fpath}')
                try:
                    st += read(fpath)
                    logging.debug('Data Loaded!')
                except Exception as error:
                    logging.warning(f"Couldn't load data file: {fname}")
                    logging.debug(f"Becuase of the following error:\n{error}")
    return st


def tr_noise_padding(tr, stime, etime, std_windows=(2, 2)):
    if isinstance(stime, float | int):
        stime = tr.stats.starttime - stime
    if isinstance(etime, float | int):
        etime = tr.stats.endtime + etime
    ###
    lst_tr = [tr]
    sps = tr.stats.sampling_rate
    ###
    sduration = (tr.stats.starttime - stime) * sps
    sduration = int(sduration)
    if sduration > 0:
        tr_std_s = tr.slice(endtime=tr.stats.starttime+std_windows[0])
        std_s = tr_std_s.std()
        snoise = np.random.normal(loc=0.0, scale=std_s, size=sduration)
        strn = Trace(snoise)
        strn.id = tr.id
        strn.stats.sampling_rate = sps
        strn.stats.starttime = tr.stats.starttime
        strn.stats.starttime -= (strn.stats.npts/sps)
        lst_tr.append(strn)
    ###
    eduration = (etime - tr.stats.endtime) * sps
    eduration = int(eduration)
    if eduration > 0:
        tr_std_e = tr.slice(starttime=tr.stats.endtime-std_windows[1])
        std_e = tr_std_e.std()
        enoise = np.random.normal(loc=0.0, scale=std_e, size=eduration)
        etrn = Trace(enoise)
        etrn.id = tr.id
        etrn.stats.sampling_rate = sps
        etrn.stats.starttime = tr.stats.endtime + 1/sps
        lst_tr.append(etrn)
    ###
    st = Stream(lst_tr)
    st.merge(-1)
    if st.get_gaps() == []:
        return st[0]
    else:
        print('There was a problem in noise-padding!')
        print(st)
        st.print_gaps()
        return None


def st_noise_padding(st, stime, etime, std_windows=(2, 2)):
    st.merge(-1)
    st.detrend('constant')
    st.merge(fill_value=0)
    st_new = Stream()
    for tr in st:
        st_new += tr_noise_padding(
            tr=tr, stime=stime, etime=etime, std_windows=std_windows
        )
    return st_new


def Coherence(stream, ref_station_id, plot=False, **kwargs):
    results = {}
    #
    tr_ref = stream.select(id=ref_station_id)[0]
    sps = tr_ref.stats.sampling_rate
    #
    for tr in stream:
        f, Cxy = scipy.signal.coherence(
            x=tr_ref.data,
            y=tr.data,
            fs=sps,
            nperseg=1024,
        )
        label = f'{tr.stats.station}.{tr.stats.channel}'
        if tr.id == ref_station_id:
            label = f'{label} (ref)'
        #
        results[label] = (f, Cxy)
    if plot:
        fig, ax = plt.subplots()
        for label, result in results.items():
            f, Cxy = result
            if 'ref' in label:
                color = 'k'
            else:
                color = None
            ax.semilogy(f, Cxy, label=label, color=color)
        ax.legend(loc=3)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Coherence')
        seisplot._finalise_figure(fig=fig, **kwargs)
    return results


def fft(array, delta, segment=None):
    '''
    array: np.array
    delta: float
    '''
    npts = array.size
    segment = segment or scipy.fftpack.helper.next_fast_len(npts)
    freq = np.fft.fftfreq(segment, d=delta)[: npts//2]
    ampl = scipy.fftpack.fft(array, segment) * delta
    ampl = np.abs(ampl[: npts//2]) / (segment*delta) # time of data = segment * delta
    return freq, ampl


class SincReconstructor:
    """
    Reconstruct signals using sinc interpolation.

    This class supports reconstruction of both NumPy arrays and ObsPy
    Stream objects. The original signal is assumed to be uniformly sampled.

    Parameters
    ----------
    target_sampling_rate : float
        Sampling rate of the reconstructed signal in Hz.

    
    Examples
    --------
        # Import required libraries
        import numpy as np
        import matplotlib.pyplot as plt
        from obspy import (
            Stream,
            Trace,
        )
        from SeisRoutine.waveform.waveform import (
            SineWaveSignal,
            SincReconstructor,
        )
        
        sampling_rate = 5
        sine_signals = SineWaveSignal(
            duration=10,
            sampling_rate=sampling_rate,
        )
        sine_signals.add_sine_wave(frequency=10, amplitude=1)
        sine_signals.add_sine_wave(frequency=20, amplitude=1)
        sine_signals.add_sine_wave(frequency=50, amplitude=1)
        time, resultant_signal = sine_signals.resultant()
        ######################################################################
        reconstructor = SincReconstructor(
            target_sampling_rate=200
        )
        ######################################################################
        reconstructed_times, reconstructed_data = (
            reconstructor.reconstruct(
                times=time,
                data=resultant_signal
            )
        )
        plt.subplots(figsize=(10, 4))
        plt.plot(time, resultant_signal, 'b', label='Original')
        plt.plot(
            reconstructed_times, reconstructed_data, 'r', label='reconstructed'
        )
        plt.legend()
        plt.show()
        ######################################################################
        trace = Trace(
            data=resultant_signal,
            header={
                "sampling_rate": 20,
                "station": "001",
                "network": "IR",
                "channel": "HHZ"
            }
        )
        stream = Stream(traces=[trace])
        
        reconstructed_stream = reconstructor.reconstruct(
            stream=stream
        )
        ## or
        # reconstructed_stream = reconstructor.reconstruct_stream(
        #     stream=stream
        # )
        
        reconstructed_stream.plot()
    """

    def __init__(self, target_sampling_rate):
        self.target_sampling_rate = target_sampling_rate

    @staticmethod
    def _sinc_wave(times, shift, nyquist_frequency):
        """
        Calculate a sinc interpolation function.

        Parameters
        ----------
        times : numpy.ndarray
            Time points where the sinc function is evaluated.
        shift : float
            Center time of the sinc function.
        nyquist_frequency : float
            Nyquist frequency of the original signal.

        Returns
        -------
        numpy.ndarray
            Sinc function values.
        """

        argument = 2 * nyquist_frequency * (times - shift)

        return np.sinc(argument)

    def reconstruct_array(self, times, data):
        """
        Reconstruct a signal represented by NumPy arrays.

        Parameters
        ----------
        times : numpy.ndarray
            Time points of the original signal.
        data : numpy.ndarray
            Amplitudes of the original signal.

        Returns
        -------
        reconstructed_times : numpy.ndarray
            Time points of the reconstructed signal.
        reconstructed_data : numpy.ndarray
            Amplitudes of the reconstructed signal.

        Raises
        ------
        ValueError
            If the input arrays have different lengths, contain fewer than
            two samples, or have a non-uniform sampling interval.

        Examples
        --------
            See the class docstring for a detailed description and usage
            examples.
        """

        times = np.asarray(times)
        data = np.asarray(data)

        if times.ndim != 1 or data.ndim != 1:
            raise ValueError("Times and data must be one-dimensional arrays.")

        if times.size != data.size:
            raise ValueError(
                "Times and data must have the same number of samples."
            )

        if times.size < 2:
            raise ValueError(
                "At least two samples are required for reconstruction."
            )

        time_intervals = np.diff(times)

        if not np.allclose(time_intervals, time_intervals[0]):
            raise ValueError(
                "The input signal must have a uniform sampling interval."
            )

        if self.target_sampling_rate <= 0:
            raise ValueError(
                "Target sampling rate must be greater than zero."
            )

        original_sampling_rate = 1 / time_intervals[0]
        nyquist_frequency = original_sampling_rate / 2

        reconstructed_times = np.arange(
            times[0],
            times[-1],
            1 / self.target_sampling_rate
        )

        reconstructed_data = np.zeros(
            reconstructed_times.shape,
            dtype=np.result_type(data, float)
        )

        for shift, amplitude in zip(times, data):
            sinc_values = self._sinc_wave(
                times=reconstructed_times,
                shift=shift,
                nyquist_frequency=nyquist_frequency
            )

            reconstructed_data += amplitude * sinc_values

        return reconstructed_times, reconstructed_data

    def reconstruct_stream(self, stream, change_station_name=True):
        """
        Reconstruct all traces in an ObsPy Stream.

        Parameters
        ----------
        stream : obspy.Stream
            Input ObsPy Stream.
        change_station_name : bool, optional
            If True, append ``"_reconst"`` to the station name.
            Defaults to True.

        Returns
        -------
        obspy.Stream
            Reconstructed ObsPy Stream.

        Examples
        --------
            See the class docstring for a detailed description and usage
            examples.
        """

        if not isinstance(stream, Stream):
            raise TypeError("Input must be an ObsPy Stream.")

        reconstructed_traces = []

        for trace in stream:
            times = trace.times()
            data = trace.data

            reconstructed_times, reconstructed_data = (
                self.reconstruct_array(
                    times=times,
                    data=data
                )
            )

            stats = trace.stats.copy()

            stats.npts = reconstructed_data.size
            stats.sampling_rate = self.target_sampling_rate
            stats.delta = 1 / self.target_sampling_rate

            if change_station_name and hasattr(stats, "station"):
                stats.station = f"{stats.station}_reconst"

            reconstructed_trace = Trace(
                data=reconstructed_data,
                header=stats
            )

            reconstructed_traces.append(reconstructed_trace)

        return Stream(reconstructed_traces)

    def reconstruct(self, data=None, times=None, stream=None):
        """
        Reconstruct either NumPy arrays or an ObsPy Stream.

        Parameters
        ----------
        data : numpy.ndarray, optional
            Amplitudes of the original signal.
        times : numpy.ndarray, optional
            Time points of the original signal.
        stream : obspy.Stream, optional
            Input ObsPy Stream.

        Returns
        -------
        tuple or obspy.Stream
            For array input, returns reconstructed times and data.
            For Stream input, returns a reconstructed Stream.

        Raises
        ------
        ValueError
            If the input arguments are invalid.

        Examples
        --------
            See the class docstring for a detailed description and usage
            examples.
        """

        if stream is not None:
            if data is not None or times is not None:
                raise ValueError(
                    "Use either stream or times/data, not both."
                )

            return self.reconstruct_stream(stream)

        if data is None or times is None:
            raise ValueError(
                "For array input, both times and data are required."
            )

        return self.reconstruct_array(
            times=times,
            data=data
        )


def transform_stream_metadata(
        st, network_mapper=None, station_mapper=None, location_mapper=None, channel_mapper=None):
    for tr in st:
        net = tr.stats.network
        sta = tr.stats.station
        loc = tr.stats.location
        cha = tr.stats.channel
        if network_mapper:
            tr.stats.network = network_mapper.get(net, net)
        if station_mapper:
            tr.stats.station = station_mapper.get(sta, sta)
        if location_mapper:
            tr.stats.location = location_mapper.get(loc, loc)
        if channel_mapper:
            tr.stats.channel = channel_mapper.get(cha, cha)
        print(net, tr.stats.network, sta, tr.stats.station, loc, tr.stats.location, cha, tr.stats.channel)


def uni_sps(st, sps=None):
    '''
    Ensures that all traces in the stream have the same sampling rate.

    Parameters:
    st (Stream): The stream of traces to check.
    sps (float, optional): The desired sampling rate. If not provided, the sampling rate of the first trace in the stream is used.

    Raises:
    AssertionError: If any trace in the stream does not have the same sampling rate as the specified or inferred sampling rate.
    '''
    sps = sps or st[0].stats.sampling_rate
    assert all(tr.stats.sampling_rate==sps for tr in st)


def preprocessing(st):
    st.merge(-1)
    st.detrend('constant')
    st.merge(fill_value=0)


class SineWaveSignal:
    """
    Generate and manage sinusoidal signals and calculate their resultant.

    Parameters
    ----------
    duration : float
        Duration of the signals in seconds.
    sampling_rate : float
        Sampling rate in samples per second (Hz).

    Attributes
    ----------
    time : numpy.ndarray
        Time vector shared by all generated signals.
    signals : list of numpy.ndarray
        List containing the generated sinusoidal signals.
    frequencies : list of float
        Frequencies of the generated signals in Hz.
    amplitudes : list of float
        Amplitudes of the generated signals.
    phases : list of float
        Phases of the generated signals in radians.

    Examples
    --------
        sine_signals = SineWaveSignal(duration=10, sampling_rate=5)
        sine_signals.add_sine_wave(frequency=10, amplitude=1)
        sine_signals.add_sine_wave(frequency=20, amplitude=1)
        sine_signals.add_sine_wave(frequency=50, amplitude=1)
        time, resultant_signal = sine_signals.resultant()
    """

    def __init__(self, duration, sampling_rate):
        self.duration = duration
        self.sampling_rate = sampling_rate

        self.time = np.arange(
            0,
            duration,
            1 / sampling_rate
        )

        self.signals = []
        self.frequencies = []
        self.amplitudes = []
        self.phases = []

    def add_sine_wave(self, frequency, amplitude, phase=0):
        """
        Generate and store a sinusoidal signal.

        Parameters
        ----------
        frequency : float
            Frequency of the sinusoidal signal in Hz.
        amplitude : float
            Amplitude of the sinusoidal signal.
        phase : float, optional
            Initial phase of the sinusoidal signal in radians.
            Default is 0.

        Returns
        -------
        numpy.ndarray
            The generated sinusoidal signal.

        Examples
        --------
        >>> signal = SineWaveSignal(
        ...     duration=10,
        ...     sampling_rate=200
        ... )

        >>> wave = signal.add_sine_wave(
        ...     frequency=10,
        ...     amplitude=1,
        ...     phase=np.pi / 4
        ... )
        """

        signal = amplitude * np.sin(
            2 * np.pi * frequency * self.time + phase
        )

        self.signals.append(signal)
        self.frequencies.append(frequency)
        self.amplitudes.append(amplitude)
        self.phases.append(phase)

        return signal

    def resultant(self):
        """
        Calculate the resultant of all stored sinusoidal signals.

        Returns
        -------
        time : numpy.ndarray
            Time vector.
        resultant_signal : numpy.ndarray
            Sum of all stored sinusoidal signals.

        Raises
        ------
        ValueError
            If no sinusoidal signal has been generated yet.
        """

        if not self.signals:
            raise ValueError("No sinusoidal signals have been generated.")

        resultant_signal = np.sum(self.signals, axis=0)

        return self.time, resultant_signal


class NoiseGenerator:
    """A modular generator for synthesized noise types with custom scaling."""

    def __init__(self, seed: int = None):
        """
        Initialize the noise generator with an optional random seed.

        Parameters:
        - seed (int, optional): Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)

    def set_seed(self, seed: int):
        """Update the random number generator seed."""
        self.rng = np.random.default_rng(seed)

    def gaussian_mad_scaled(
        self,
        length: int,
        target_mad: float,
        scale_factor: float = 1.0,
    ) -> np.ndarray:
        """
        Generate Gaussian noise calibrated to match a target Median Absolute
        Deviation (MAD), scaled by a custom factor.

        Parameters:
        - length (int): Number of noise samples to generate.
        - target_mad (float): Target MAD of the unscaled noise.
        - scale_factor (float): Multiplier applied to the calibrated noise.

        Returns:
        - np.ndarray: Scaled 1D array of Gaussian noise.
        
        Examples:
        --------
            ng = NoiseGenerator(seed=43)

            # Generate noise matching MAD = 2.0 with scale factor = 1.0
            noise = ng.gaussian_mad_scaled(
                length=int(1e6),
                target_mad=2.0,
                scale_factor=2.0,
            )

            # Verify the result
            measured_mad = scipy.stats.median_abs_deviation(noise)
            print(f"Target MAD: 2.0 | Measured MAD: {measured_mad}")
            # Output: Target MAD: 2.0 | Measured MAD: 2.0
        """

        # Theoretical conversion factor:
        # MAD = sigma * norm.ppf(0.75) ≈ 0.67449 * sigma
        normal_mad_ratio = scipy.stats.norm.ppf(0.75)
        target_sigma = target_mad / normal_mad_ratio

        # Generate zero-mean Gaussian noise using the instance generator
        raw_noise = self.rng.normal(
            loc=0.0,
            scale=target_sigma,
            size=length,
        )

        # Center around median to eliminate sample bias
        raw_noise = raw_noise - np.median(raw_noise)

        # Empirically calibrate to guarantee exact MAD match
        empirical_mad = scipy.stats.median_abs_deviation(raw_noise)
        if empirical_mad > 0:
            base_noise = raw_noise * (target_mad / empirical_mad)
        else:
            base_noise = raw_noise

        return scale_factor * base_noise
