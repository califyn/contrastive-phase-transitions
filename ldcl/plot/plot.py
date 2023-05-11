import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np
import os

from scipy.signal import savgol_filter

from .color import get_cmap


def pretty_print(obj):
            if isinstance(obj, float):
                return np.format_float_positional(obj, precision=4)
            else:
                return obj

def plot_loss(metric, title, save_progress_path, xlabel = 'Epochs', ylabel = 'Loss', save_name = 'training_loss.png'):
    metric = savgol_filter(metric, 51, 3)
    plt.figure()
    plt.plot(np.arange(metric.shape[0]), metric)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.savefig(os.path.join(save_progress_path, save_name))

def plot_metric(metrics, title, save_progress_path, xlabel = 'Epochs', ylabel = 'Loss', save_name = 'training_loss', legend = None):
    plt.figure()

    for metric in metrics:
        if isinstance(metric, list): metric = np.array(metric)
        plt.plot(np.arange(metric.shape[0]), metric)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        if legend: plt.legend(legend)

        plt.savefig(os.path.join(save_progress_path, save_name + '.png'))
    

    # for metric in metrics:
    #     metric = savgol_filter(metric, 51, 3)
    #     plt.plot(np.arange(metric.shape[0]), metric)
    #     plt.title(title)
    #     plt.xlabel(xlabel)
    #     plt.ylabel(ylabel)

    #     if legend: plt.legend(legend)

    #     plt.savefig(os.path.join(save_progress_path, save_name + '_savgol' + '.png'))


class VisPlot:
    """
        Plot class for easy plotting!

        Initialize like this, specifying the number of dimensions (MPL 2d or Plotly 3d)
        and then the number of subplots (optional if one):

            plot = VisPlot(2, num_subplots=2)

        In general, the syntax is identical for 2D or 3D plots, so you can switch
        easily in between them.

        Plots can be successively added. This means you can plot multiple datasets at
        once, although be careful to use the same dimensionality reduction on all of
        them if you reduce dimensions. The most simple command is VisPlot.add:

            points = np.random.rand(100,2)
            label = {"num": np.array(list(range(100)))}
            a_color = plt.get_cmap("viridis")(label["num"] / 100)
            b_color = plt.get_cmap("plasma")(label["num"] / 100)
            plot.add(points, size=np.linspace(1,2,num=100), color=np.stack(a_color, b_color), label=label)

        Of course, it would be more correct to use ldcl.color's get_cmap and not
        plt's. This would plot 100 random points in growing size with matplotlib in 2d,
        with two subplots, identical except the left one uses viridis colormap and the
        right one uses plasma colormap. Note that the default behavior is to plot
        the same points but with different colors.

        If you ran this code again, you'd get 200 such points. This means that you can
        plot a bunch of things at once, on the same axis, in an accumulative
        fashion.

        There is also a convenience function (add_with_cmap) that lets you specify
        the color using a combination of a colormap and once of the values in the
        labels dictionary. This is probably more convenient for everyday plotting, but
        if you want more control over the coloring, add will give you the most control.
        An example of how to use add_with_cmap to do what we just did would
        look like this:

            plot.add_with_cmap(points, label, cmap=["viridis", "plasma"], cby=["num", "num"], size=np.linspace(1, 2, num=100))

        Note that ldcl.color's colormaps automatically normalize data so label["num"]
        does not need to be normalized.

        To plot different points across subplots, specify the subplot option when using
        VisPlot.add. This trickles down to add_with_cmap, which uses VisPlot.add
        internally.

        There is also a set_title function to automatically set all the titles of the
        subplot(s). Its usage is pretty straightforward:

            plot.set_title(["viridis", "plasma"])

        in the previous instance.

        To do more dramatic changes to the figure (e.g. add labels, axis titles, etc.)
        it is best to access the internal objects VisPlot.fig (which is the figure in
        both MPL and Plotly) and VisPlot.axs, which in MPL mode is a list of all the
        Axes objects of the different plots. I find it profoundly annoying that you
        will probably mostly interact with VisPlot.axs in MPL mode and VisPlot.fig
        in Plotly mode but I can't think of any workaround for now.

        Finally, to show the plot, simply call

            plot.show()

        For Plotly, it writes it to plot.html in the same directory and then opens it.
        For MPL, it just does the interactive widget there where it opesn up its own
        app.

        To do this exact same demo but in 3d, replace

            plot = VisPlot(2, num_subplots=2) => plot = VisPlot(3, num_subplots=2)
            points = np.random.rand(100,2) => points = np.random.rand(100,3)
    """

    def __init__(self, dims, num_subplots=1):
        """
            Create a figure object in order to sequentially add plots.

            :param dims: the number of dimensions (mpl for 2, plotly for 3).
            :param num_subplots: number of desired subplots.
        """

        if dims == 2:
            self.mode = "mpl_2d"

            fig = plt.figure()
            axs = []
            for i in range(num_subplots):
                axs.append(fig.add_subplot(1, num_subplots, i + 1))
        elif dims == 3:
            self.mode = "plotly_3d"

            camera = dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0.5, y=1.25, z=0)
            )

            fig = make_subplots(rows=1, cols=num_subplots, specs = [[{'is_3d':True}] * num_subplots], subplot_titles=[" "] * num_subplots)

            fig.update_layout(scene_camera=camera)
            fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)

        self.fig = fig
        self.dims = dims
        self.num_subplots = num_subplots

        if dims == 2:
            self.axs = axs

    def add(self, points, size=None, color=None, label=None, outline=False, subplot=None):
        """
            Plots some points onto the figure (it's successively added).

            :param points: Nxdims points to be plotted.
            :param size: arr of length N, or scalar, that specifies size of points, default all 1.
            :param color: arr of size {1, num_subplots} x N x {3,4} that specify colors, default all black. The first dim is 1 only if num_subplots is 1, or if subplot is specified.
            :param outline: (3D only) True if you want to see an outline around the points.
            :param label: dictionary to use as labels, default no additional labels.
            :param subplot: which subplot(s) to plot. Only use if you plotting different points across
                different subplots, otherwise don't use it and just plot all the subplots at once.
        """
        if isinstance(subplot, int):
            subplot = [subplot]

        if len(np.shape(color)) == 2:
            lcolor = color[np.newaxis, :, :]
        else:
            lcolor = color

        num_to_plot = self.num_subplots if subplot == None else len(subplot)
        assert(np.shape(lcolor)[0] == num_to_plot) # check number of colors is eq to num to plot
        assert(np.shape(points)[1] == self.dims) # check that the number of dimensions of embeddings is correct

        if size is None:
            size = 1
        if color is None:
            color = np.array([0,0,0])
        if label is None:
            label = {}

        outline_width = 1 if outline else 0

        plot_it = subplot if subplot != None else range(self.num_subplots) # the plots to plot
        for plot in plot_it:
            if self.mode == "mpl_2d":
                self.axs[plot].scatter(points[:, 0], points[:, 1], s=size, c=lcolor[plot])
            elif self.mode == "plotly_3d":
                self.fig.add_trace(go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=size,
                        color=lcolor[plot],
                        line=dict(width=outline_width, color='black'),
                        opacity=1
                    ),
                    text=["<br>".join([f"{k}={pretty_print(v[i])}" for k, v in label.items()]) for i in range(np.shape(points)[0])]
                ), 1, plot + 1)

    def add_with_cmap(self, points, label, cmap=None, cby=None, **otherargs):
        """
            Even faster plotting, like VisPlot.add but you can specify the
            cmaps and variables to color by immediately.

            Example usage:
                VisPlot.add_with_cmap(points, label, cmap=["viridis", "viridis"], cby=["x", "y"], outline=False)

            :param points: the points you want to plot (see VisPlot.add)
            :param label: dictionary containing relevant information about the points,
                and which the colors will come from.
            :param cmap: a single colormap function, a string specifying one, or multiple
                (as many as num_subplots) as a list which will map the indicated variables.
                More explanation at cby.
            :param cby: the "colored-by" variables, which you can retrieve as keys
                from label. The nth variable is colored using the nth colormap, so cby
                and cmap should have the same length, VisPlot.num_subplots. The only
                exception is if you want the same colormap for all variables, then you
                can have cmap length 1.
            :otherargs: all the other keyword arguments used to plot, see
                VisPlot.add. They can be used here (note: identical across all subplots).
        """
        if cby == None: # default
            cby = list(label.keys())
        if isinstance(cby, str): # fix if length 1
            cby = [cby]

        if cmap == None: # default
            cmap = ["viridis"] * len(cby)
        if callable(cmap) or isinstance(cmap, str): # fix if length 1
            cmap = [cmap]
        for idx, m in enumerate(cmap): # make sure they're callable
            if isinstance(m, str):
                cmap[idx] = get_cmap(m)
        if len(cmap) == 1 and self.num_subplots > 1: # tile if there's only one
            cmap = [cmap[0]] * self.num_subplots

        assert(len(cby) == self.num_subplots)

        color = np.array([col(label[var]) for var, col in zip(cby, cmap)])
        self.add(points, color=color, label=label, **otherargs)
        self.set_title(cby)

    def set_title(self, titles):
        """
            Set titles for the plots.

            :param titles: one string or a list of strings, of length
                self.num_subplots, which are the new plot titles.
        """
        if isinstance(titles, str): # fix if has length 1
            titles = [titles]

        assert(len(titles) == self.num_subplots)

        if self.mode == "mpl_2d":
            for it, title in enumerate(titles):
                self.axs[it].set_title(title)
        elif self.mode == "plotly_3d":
            for it, title in enumerate(titles):
                self.fig.layout.annotations[it].update(text=title)

    def show(self):
        """
            Displays the resulting figure.
        """

        if self.mode == "mpl_2d":
            plt.show()
        elif self.mode == "plotly_3d":
            self.fig.write_html('plot.html', auto_open=True)



