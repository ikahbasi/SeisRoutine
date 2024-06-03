import os
import matplotlib.pyplot as plt
from obspy.imaging.cm import pqlx
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import SeisRoutine.core as src
import numpy as np
import math
import inspect
import seaborn as sns


def _get_proper_kwargs(func: object, kwargs: dict):
    sig = inspect.signature(func)
    kw = {k: v for k, v in kwargs.items()
          if k in sig.parameters.keys()}
    return kw


def _finalise_ax(ax,
                 xlabel: str=None, ylabel: str=None,
                 xlim: list=None, ylim: list=None,
                 labelsize: int=10, linewidth: int=2,
                 grid: bool=False, **kwargs):
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
    figsize = kwargs.get("figsize", (10.5, 7.5))

    fig.set_size_inches(figsize)

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
                          ax=None,
                          **kwargs):
    '''
    Doc ???
    '''
    if ax is None:
        fig, ax = plt.subplots()
    #
    xcenters, ycenters, z = src.density_meshgrid(
        x=x, y=y, xstep=xstep, ystep=ystep, zreplace=0.9
    )
    # Get a proper kwargs for the plt.pcolormesh function.
    kw = _get_proper_kwargs(func=plt.pcolormesh, kwargs=kwargs)
    im = ax.pcolormesh(
        xcenters, ycenters, z,
        cmap=pqlx, shading='gouraud', **kw
    )
    cbaxes = ax.inset_axes(
        bounds=[0.8, 0.94, 0.15, 0.03],
        transform=ax.transAxes
    )
    cbar = ax.figure.colorbar(im, cax=cbaxes, orientation='horizontal')
    cbaxes.xaxis.set_ticks_position("bottom")
    cbar.ax.set_xlabel('Counts')
    #
    kw = _get_proper_kwargs(
        func=_finalise_ax, kwargs=kwargs)
    _finalise_ax(ax, **kw)
    #
    _finalise_figure(ax.figure, **kwargs)


def histogram(arr, step=0.5, log=True,
              ax=None, fig=None, **kwargs):
    '''
    Docs ???
    '''
    if ax is None:
        fig, ax = plt.subplots()
    #
    bins_min = 0
    while bins_min > min(arr):
        bins_min -= step
    bins_max = math.ceil(max(arr)) + step
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
    _finalise_figure(ax.figure, **kwargs)


def density_hist(x: np.array, y: np.array,
                 xstep: float, ystep: float,
                 kind: str='density', histlog: bool=True,
                 axes: object=None,
                 **kwargs):
    #
    file_management = {'save': kwargs.get('save', False),
                       'show': kwargs.get('show', False)}
    kwargs.update(save=False, show=False)
    if axes is None:
        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=(12, 6),
            sharey='row',# sharex='col',
            gridspec_kw={'width_ratios': [5, 1]})
        plt.subplots_adjust(bottom=0.15, hspace=0, wspace=0)
    else:
        (ax1, ax2) = axes
    #
    if kind == 'scatter':
        kw = _get_proper_kwargs(
            func=sns.scatterplot, kwargs=kwargs
        )
        sns.scatterplot(
            x=x, y=y,
            alpha=0.4, s=20, color='black',
            ax=ax1, **kw
        )
    elif kind == 'density':
        plot_density_meshgrid(
            x, y,
            xstep=xstep, ystep=ystep,
            ax=ax1, **kwargs
        )
    kw = _get_proper_kwargs(func=_finalise_ax, kwargs=kwargs)
    _finalise_ax(ax1, **kw)
    #
    for key in ['ylabel', 'xlim', 'ylim']:
        _ = kwargs.pop(key, None)
    kwargs['xlabel'] =  'Abundance'
    histogram(
        arr=y,
        step=ystep, log=histlog,
        ax=ax2, fig=fig,
        **kwargs
    )
    kwargs.update(file_management)
    _finalise_figure(ax1.figure, **kwargs)
