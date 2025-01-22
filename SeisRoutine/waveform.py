import numpy as np
import scipy
from obspy import Stream, Trace
from scipy import signal
import matplotlib.pyplot as plt
import SeisRoutine.plot as seisplot


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
