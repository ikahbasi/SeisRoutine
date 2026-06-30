import numpy as np
import scipy
from obspy import Stream, Trace
from obspy import read
from scipy import signal
import matplotlib.pyplot as plt
import SeisRoutine.plot as seisplot
import SeisRoutine.core as src
import logging
import re
import os
import glob
import obspy as obs
from scipy import stats
import pywt
from scipy.signal import find_peaks
from dataclasses import dataclass


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
        z_score = stats.zscore(self.data)
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
        peaks, properties = find_peaks(
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
        s = stats.skew(data, bias=False)
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
        k = stats.kurtosis(data, fisher=fisher, bias=False)
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

        self.data = self.data - self.data.mean(axis=1, keepdims=True)

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

    @staticmethod
    def _compute_power(
            data,
            axis=1,
            domain='time',
        ):
        
        n = data.shape[axis]

        if domain == 'time':
            power = 1 / n * np.sum(np.abs(data) ** 2, axis=axis)

        elif domain == 'frequency':
            power = 1 / (n ** 2) * np.sum(np.abs(data) ** 2, axis=axis)

        else:
            raise ValueError("domain must be 'time' or 'frequency'")
        
        return power

    def power_in_time(
            self,
            epsilon=1e-8,
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
        ) + epsilon

        return p_signal / p_noise

    def power_in_freq(
            self,
            epsilon=1e-8,
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
        ) + epsilon

        return p_signal / p_noise

    def mad(
            self,
        ):

        noise_mad = stats.median_abs_deviation(self.noise, axis=1)
        signal_mad = stats.median_abs_deviation(self.signal, axis=1)

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
                noise_p = 1.4826 * stats.median_abs_deviation(noise)

            elif method == 2:

                signal = signal[
                    (signal >= np.percentile(signal, lbp)) &
                    (signal <= np.percentile(signal, hbp))
                ]
                signal_p = stats.median_abs_deviation(signal)

                noise = noise[
                    (noise >= np.percentile(noise, lbp)) &
                    (noise <= np.percentile(noise, hbp))
                ]
                noise_p = stats.median_abs_deviation(noise)

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
            root: str,
            pattern_path: str,
            merge_method=None,
            **pattern_vars
        ):
        self.root = root
        self.pattern_path = pattern_path
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

    def get_by_pick(self, pick):
        """Convenience wrapper that extracts time and station from a Pick."""
        return self.get(
            time=pick.time,
            station_code=pick.waveform_id.station_code
        )
    
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
        pattern = self.pattern_path.format(time=time, **self.pattern_vars)
        pattern_path = f"{self.root}/{pattern}"
        logging.info(f"Reading waveform data: {pattern_path}")

        self.stream = self._read_safely(pattern_path)
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
        f, Cxy = signal.coherence(x=tr_ref.data, y=tr.data, fs=sps, nperseg=1024)
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


class reconstruction:
    def __init__(self, st):
        self.st = st
        self.st_reconstructed = Stream()

    def _sinc_wave(self, frequency, duration, sampling_rate, shift):
        stime = -duration / 2
        etime =  duration / 2
        delta = 1 / sampling_rate
        times = np.arange(stime, etime, delta)
        times_shifted = times + shift
        x = 2 * np.pi * frequency * times_shifted
        ampls = np.sin(x) / (x)
        return times, ampls

    def _reconstruction(self, times, data, target_sps):
        delta = times[1] - times[0]
        sampling_rate = 1 / delta
        nyquest_frequency = sampling_rate / 2
        duration = times[-1] - times[0] + delta
        #
        t_reconstructed = np.arange(0, duration, 1/target_sps)
        a_reconstructed = np.zeros_like(t_reconstructed)
        for shift, scale in zip(times, data):
            t_sinc, a_sinc = self._sinc_wave(
                frequency=nyquest_frequency,
                duration=duration,
                sampling_rate=target_sps,
                shift=(duration/2)-shift
                )
            a_reconstructed += a_sinc * scale
        return t_reconstructed, a_reconstructed
    
    def apply(self, target_sps):
        self.st_reconstructed = Stream()
        for tr in self.st:
            times = tr.times()
            data = tr.data
            ###
            t_reconstructed, a_reconstructed = self._reconstruction(times, data, target_sps)
            stats_reconstructed = tr.stats.copy()
            stats_reconstructed.npts = a_reconstructed.size
            stats_reconstructed.delta = t_reconstructed[1] - t_reconstructed[0]
            tr_reconstructed = Trace(data=a_reconstructed, header=stats_reconstructed)
            self.st_reconstructed += tr_reconstructed


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


def reconstrucion(stream, target_sps, change_name=True):
    """
    Reconstruct an ObsPy Stream using sinc interpolation.

    This function reconstructs each Trace within an ObsPy Stream to a new sampling rate
    using sinc interpolation.

    Parameters:
        stream (obspy.core.stream.Stream): The input ObsPy Stream object.
        target_sps (int): The target sampling rate for the reconstructed Traces.
        change_name (bool, optional): If True, appends '_reconst' to the station name of
            each Trace. Defaults to True.

    Returns:
        obspy.core.stream.Stream: The reconstructed ObsPy Stream object.
    """
    lst_trace = []
    for trace in stream:
        times, data = src.reconstrucion(times=trace.times(),
                                        amplitudes=trace.data,
                                        target_sps=target_sps)
        # Copy the stats from the original Trace and update the stats
        # with the new number of points and sampling rate.
        stats = trace.stats.copy()
        stats.npts = data.size
        stats.sampling_rate = target_sps
        if change_name:
            stats.station += '_reconst'
        # Create a new Trace object with the reconstructed data and updated stats.
        trace_reconst = Trace(data=data, header=stats)
        lst_trace.append(trace_reconst)
    # Create a new Stream object from the list of reconstructed Traces.
    stream_reconst = Stream(lst_trace)
    return stream_reconst
