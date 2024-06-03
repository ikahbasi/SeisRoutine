import os
import matplotlib.pyplot as plt
from obspy.imaging.cm import pqlx
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import SeisRoutine.core as src
import numpy as np
import math
import inspect


def _get_proper_kwargs(func: object, kwargs: dict):
    sig = inspect.signature(func)
    kw = {k: v for k, v in kwargs.items()
          if k in sig.parameters.keys()}
    return kw


def _finalise_ax(ax,
                 xlabel: str=None, ylabel: str=None,
                 xlim: list=None, ylim: list=None,
                 labelsize: int=10, linewidth: int=2,
                 grid: bool=False):
    """
    Internal function to wrap up an ax.
    {plotting_kwargs}
    """
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    #
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #
    ax.xaxis.set_tick_params(labelsize=labelsize)
    ax.yaxis.set_tick_params(labelsize=labelsize)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(linewidth)
    if grid:
        ax.grid()



def _finalise_figure(fig, **kwargs):
    """
    Internal function to wrap up a figure.
    {plotting_kwargs}
    """
    show = kwargs.get("show", True)
    save = kwargs.get("save", False)
    savefile = kwargs.get("savefile", "figure.png")
    title = kwargs.get("title")
    return_fig = kwargs.get("return_figure", False)
    size = kwargs.get("size", (10.5, 7.5))

    fig.set_size_inches(size)

    if title:
        plt.title(title, fontsize=25)
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


def plot_density_meshgrid(x: np.array, y: np.array,
                          xstep: float, ystep: float,
                          ax=None, fig=None,
                          **kwargs):
    '''
    Doc ???
    '''
    xcenters, ycenters, z = src.density_meshgrid(
        x=x, y=y, xstep=xstep, ystep=ystep, zreplace=0.9)
    ###
    if ax is None:
        fig, ax = plt.subplots()
    # Get a proper kwargs for the plt.pcolormesh function.
    kw = _get_proper_kwargs(
        func=plt.pcolormesh, kwargs=kwargs)
    im = ax.pcolormesh(xcenters, ycenters, z,
                       cmap=pqlx,
                       shading='gouraud', **kw)
    #
    cbaxes = ax.inset_axes(
        bounds=[0.8, 0.94, 0.15, 0.03],
        transform=ax.transAxes
        )
    cbar = fig.colorbar(im, cax=cbaxes, orientation='horizontal')
    cbaxes.xaxis.set_ticks_position("bottom")
    cbar.ax.set_xlabel('Counts')
    #
    kw = _get_proper_kwargs(
        func=_finalise_ax, kwargs=kwargs)
    _finalise_ax(ax, **kw)
    #
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
    #
    kw = _get_proper_kwargs(
        func=_finalise_ax, kwargs=kwargs)
    _finalise_ax(ax, **kw)
    _finalise_figure(fig, **kwargs)
