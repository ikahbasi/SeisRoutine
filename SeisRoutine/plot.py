import os
import matplotlib.pyplot as plt


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
        plt.ylim(top=ylim[1])
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
