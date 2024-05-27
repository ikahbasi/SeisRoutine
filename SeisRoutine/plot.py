import os
import matplotlib.pyplot as plt
from obspy.imaging.cm import pqlx
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import SeisRoutine.core as src
import numpy as np
import math


def _finalise_figure(fig, **kwargs):
    """
    Internal function to wrap up a figure.
    {plotting_kwargs}
    """
    show = kwargs.get("show", True)
    save = kwargs.get("save", False)
    savefile = kwargs.get("savefile", "figure.png")
    title = kwargs.get("title")
    xlim = kwargs.get("xlim")
    ylim = kwargs.get("ylim")
    return_fig = kwargs.get("return_figure", False)
    size = kwargs.get("size", (10.5, 7.5))

    fig.set_size_inches(size)
    for ax in fig.axes:
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)

    if title:
        plt.title(title, fontsize=25)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    if save:
        path = os.path.dirname(savefile)
        if path:
            os.makedirs(path, exist_ok=True)
        fig.savefig(savefile, bbox_inches="tight", dpi=130)
        print("Saved figure to {0}".format(savefile))
    if show:
        plt.show(block=True)
    if return_fig:
        return fig
    else:
        return None


def plot_density_meshgrid(x, y,
                          xstep, ystep,
                          xlabel='Distance [km]',
                          ylabel='Time Difference',
                          ax=None, fig=None,
                          show_cmap=True, norm='log',
                          **kwargs):
    '''
    Doc ???
    '''
    xcenters, ycenters, z = src.density_meshgrid(
        x=x, y=y, xstep=xstep, ystep=ystep, zreplace=0.9)
    ###
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.pcolormesh(xcenters, ycenters, z,
                       cmap=pqlx,
                       shading='gouraud', norm=norm)
    if show_cmap:
        cbaxes = inset_axes(ax, width="20%", height="2%", loc=1,
                            bbox_to_anchor=(-0.02, 0., 1, 1),
                            bbox_transform=ax.transAxes,)
        cbar = fig.colorbar(im, cax=cbaxes, orientation='horizontal')
        cbaxes.xaxis.set_ticks_position("bottom")
        cbar.ax.set_xlabel('Counts')
    #
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    _finalise_figure(fig, **kwargs)


def histogram(arr, step=0.5, log=True,
              ax=None, fig=None, **kwargs):
    '''
    Docs ???
    '''
    bins_min = math.ceil(min(arr))
    while bins_min > min(arr):
        bins_min -= step
    bins_max = math.ceil(max(arr)) + step
    if ax is None:
        fig, ax = plt.subplots()
    #
    bins = np.arange(bins_min, bins_max, step)
    bins -= step/2
    ax.hist(arr, bins=bins,
            alpha=1, edgecolor='k', facecolor='g',
            orientation='horizontal', log=log)
    _finalise_figure(fig, **kwargs)
