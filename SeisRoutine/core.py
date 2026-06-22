import numpy as np
import latlon as ll


class SeisanCoord:
    """
    Convert coordinates between Seisan/HYPO71 and Decimal Degree formats.

    Seisan format:  2825.46N  (DDmm.mmH)
    DD format:      28.4243

    Examples:
        >>> lat = SeisanCoord('2825.46N')
        >>> lat.dd
        28.4243
        >>> lat.seisan
        '2825.46N'

        >>> lon = SeisanCoord(51.917, is_longitude=True)
        >>> lon.dd
        51.917
        >>> lon.seisan
        '05155.02E'

        >>> SeisanCoord.from_seisan('2825.46S').dd
        -28.4243

        >>> SeisanCoord.from_dd(-100.917, is_longitude=True).seisan
        '10055.02W'
    """

    def __init__(self, value: float | str, is_longitude: bool = False):
        """
        Args:
            value:
                Decimal degree (float) or Seisan string (e.g. '2825.46N').
            is_longitude:
                Required only when value is a float. Ignored for strings.
        """
        if isinstance(value, str):
            self._dd, self._is_longitude = self._parse_seisan(value)
        elif isinstance(value, (int, float)):
            self._dd = float(value)
            self._is_longitude = is_longitude
        else:
            msg = "value must be a Seisan string or a numeric decimal degree."
            raise TypeError(msg)

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------

    @property
    def dd(self) -> float:
        """Coordinate in Decimal Degree format."""
        return round(self._dd, 6)

    @property
    def seisan(self) -> str:
        """Coordinate in Seisan/HYPO71 string format."""
        return self._to_seisan(self._dd, self._is_longitude)

    @property
    def is_longitude(self) -> bool:
        """
        True if the coordinate represents longitude (E/W),
        False for latitude (N/S).
        """
        return self._is_longitude

    # ------------------------------------------------------------------
    #  Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_seisan(coord_str: str) -> tuple[float, bool]:
        coord_str = coord_str.strip().upper()
        hemisphere = coord_str[-1]

        if hemisphere not in ('N', 'S', 'E', 'W'):
            msg = f"Invalid hemisphere '{hemisphere}'. Must be N, S, E, or W."
            raise ValueError(msg)

        is_longitude = hemisphere in ('E', 'W')
        degree_digits = 3 if is_longitude else 2
        numeric = coord_str[:-1]

        degrees = float(numeric[:degree_digits])
        minutes = float(numeric[degree_digits:])
        dd = degrees + minutes / 60.0

        if hemisphere in ('S', 'W'):
            dd = -dd

        return dd, is_longitude

    @staticmethod
    def _to_seisan(dd: float, is_longitude: bool) -> str:
        if is_longitude:
            hemisphere = ('E' if dd >= 0 else 'W')
        else:
            hemisphere = ('N' if dd >= 0 else 'S')
        abs_dd = abs(dd)
        degrees = int(abs_dd)
        minutes = (abs_dd - degrees) * 60.0

        if is_longitude:
            return f"{degrees:03d}{minutes:05.2f}{hemisphere}"
        else:
            return f"{degrees:02d}{minutes:05.2f}{hemisphere}"

    # ------------------------------------------------------------------
    #  Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_dd(cls, dd: float, is_longitude: bool = False) -> 'SeisanCoord':
        """
        Create an instance from a Decimal Degree value.
        """

        return cls(dd, is_longitude=is_longitude)

    @classmethod
    def from_seisan(cls, coord_str: str) -> 'SeisanCoord':
        """
        Create an instance from a Seisan/HYPO71 coordinate string.
        """
    
        return cls(coord_str)

    # ------------------------------------------------------------------
    #  Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        kind = "Lon" if self._is_longitude else "Lat"
        return f"SeisanCoord({kind}: dd={self.dd}, seisan='{self.seisan}')"

    def __float__(self) -> float:
        return self._dd

    def __str__(self) -> str:
        return self.seisan


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
