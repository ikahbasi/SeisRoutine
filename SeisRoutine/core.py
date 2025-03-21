import numpy as np
import latlon as ll

def making_latlon(coord_str: str='5 52 59.88 N',
                  format: str='d% %m% %S% %H'):
    '''
    convert coordinate: Degrees°minutes'seconds" ---> Decimal.degrees
                        22°45'45                 ---> 22.7625
    '''
    hemisphere = coord_str[-1]
    #
    if hemisphere in ['E', 'W']:
        coord_class = ll.Longitude
    elif hemisphere in ['N', 'S']:
        coord_class = ll.Latitude
    else:
        raise ValueError('Hemisphere identifier N, S, E or W')
    #
    coord = ll.string2geocoord(coord_str, coord_class, format)
    return coord


def dm2dd(coord_str: str):
    '''
    convert coordinate: Degree-Minute ---> Decimal-Degree
                        2245.45N      ---> 22.7575
    The Degree-Minute format used by HYPO71 and Seisan
    in STATION0.HYP file.
    '''
    #Parsing the components of a coordinate string.
    degree = coord_str[0: 2]
    minute = coord_str[2: -1]
    hemisphere = coord_str[-1]
    # Reformat the string of the input coordinate.
    coord_str = f'{degree} {minute} {hemisphere}'
    format = 'd% %M% %H'
    #
    coord = making_latlon(coord_str, format)
    return coord.decimal_degree


def density_meshgrid(x, y, xstep, ystep, zreplace=0.9):
    '''
    Docs ???
    '''
    ymin = min(y) - (ystep/2)
    ymax = max(y) + (ystep*2)
    xmin = min(x) - (xstep/2)
    xmax = max(x) + (xstep*2)
    #
    bins_y = np.arange(ymin, ymax, ystep)
    bins_x = np.arange(xmin, xmax, xstep)
    #
    hight, xedges, yedges = np.histogram2d(
        x, y, bins=(bins_x, bins_y), density=False
        )
    #
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    z = hight.T
    #
    z[z == 0] = zreplace
    return xcenters, ycenters, z


# def limited_hist(data, bins_range=(-5, 5, 0.5), bins_type='center'):
#     '''
#     Doc ???
#     '''
#     bins = np.arange(*bins_range)
#     bin_min, bin_max, bin_step = bins_range
#     if bins_type == 'center':
#         bins = bins - (bin_step/2)
#     bin_labels = bins.copy()
#     ###
#     if data.min() < bins[0]:
#         data[data < bins[0]] = bins[0] + (bin_step/4)
#         bin_labels[0] = data.min()
#     if bins[-1] < data.max():
#         data[data > bins[-1]] = bins[-1] - (bin_step/4)
#         bin_labels[-1] = data.max()
#     ###
#     hist, bin_edges = np.histogram(data, bins=bins)
#     return hist, bin_edges, bin_labels

# def limited_hist(data, bins_range=(-5, 5, 0.5), bins_type='center'):
#     hist, bin_edges, bin_labels = limited_hist(data, bins_range, bins_type)
#     fig = plt.figure(figsize=(10, 5))
#     rects = plt.bar(bin_edges[:-1], )
#     rects = plt.hist(data, bins, align='mid', edgecolor='black', linewidth=1.2, label=label)
#     autolabel(rects)
#     plt.ylabel('Abundance [count]', fontsize=17)
#     plt.xlabel('RMS [s]', fontsize=17)
#     plt.xticks(bins, bins_label)
#     plt.title(title)
#     #fig = _finalise_figure(fig=fig, **kwargs)


# def ResidualHistogramVertical(arr, ax, ylim=[-5, 5], ystep=0.5):
#     bins = np.arange(ylim[0]+ystep/2, ylim[1], ystep)
#     bins[0] = ylim[0]
#     bins[-1] = ylim[1]
#     ax.hist(arr, bins=bins,
#             alpha=1, edgecolor='k', facecolor='g',
#             orientation='horizontal', log=False)


def sinc_wave(times, shift, frequency):
    """
    Generate a sinc wave.

    Parameters:
        times (np.array): Array of time points.
        shift (int): Time shift of the wave.
        frequency (int): Frequency of the wave.

    Returns:
        np.array: Sinc wave values at the given time points.
    """
    t = 2 * frequency * (times - shift)
    return np.sinc(t)


def reconstrucion(times, amplitudes, target_sps):
    """
    Reconstruct a signal using sinc interpolation.

    This function reconstructs a signal from a set of discrete time points and amplitudes
    to a new set of time points with a specified target sampling rate.

    Parameters:
        times (np.array): Array of time points for the original signal.
        amplitudes (np.array): Array of amplitudes corresponding to the original time points.
        target_sps (int): Target sampling rate for the reconstructed signal.

    Returns:
        tuple: A tuple containing:
            - np.array: Array of time points for the reconstructed signal.
            - np.array: Array of amplitudes for the reconstructed signal.
    """
    # Calculate the parameters of the original signal
    stime = times[0]
    etime = times[-1]
    delta = times[1] - times[0]
    sampling_rate = 1 / delta
    nyquest_frequency = sampling_rate / 2
    # Initialize arrays to store time points and amplitudes of the reconstructed signal.
    times_recunstructed = np.arange(stime, etime, 1/target_sps)
    amplitudes_reconstructed = np.zeros_like(times_recunstructed)
    # Loop to compute the reconstructed amplitudes.
    for shift, scale in zip(times, amplitudes):
        a_sinc = sinc_wave(frequency=nyquest_frequency,
                           times=times_recunstructed,
                           shift=shift)
        amplitudes_reconstructed += scale * a_sinc
    return times_recunstructed, amplitudes_reconstructed
