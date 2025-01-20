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


def fft(array, delta):
    '''
    array: np.array
    delta: float
    '''
    npts = array.size
    segment = scipy.fftpack.helper.next_fast_len(npts)
    freq = np.fft.fftfreq(segment, d=delta)[: npts//2]
    ampl = scipy.fftpack.fft(array, segment) * delta
    ampl = np.abs(ampl[: npts//2]) / (segment*delta) # time of data = segment * delta
    return freq, ampl