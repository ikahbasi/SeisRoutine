import numpy as np


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


def limited_hist(data, bins_range=(-5, 5, 0.5), bins_type='center'):
    '''
    Doc ???
    '''
    bins = np.arange(*bins_range)
    bin_min, bin_max, bin_step = bins_range
    if bins_type == 'center':
        bins = bins - (bin_step/2)
    bin_labels = bins.copy()
    ###
    if data.min() < bins[0]:
        data[data < bins[0]] = bins[0] + (bin_step/4)
        bin_labels[0] = data.min()
    if bins[-1] < data.max():
        data[data > bins[-1]] = bins[-1] - (bin_step/4)
        bin_labels[-1] = data.max()
    ###
    hist, bin_edges = np.histogram(data, bins=bins)
    return hist, bin_edges, bin_labels

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


def ResidualHistogramVertical(arr, ax, ylim=[-5, 5], ystep=0.5):
    bins = np.arange(ylim[0]+ystep/2, ylim[1], ystep)
    bins[0] = ylim[0]
    bins[-1] = ylim[1]
    ax.hist(arr, bins=bins,
            alpha=1, edgecolor='k', facecolor='g',
            orientation='horizontal', log=False)