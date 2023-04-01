import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_cmap(cmap, normalize=True):
    """
        Easily retrieve colormaps from matplotlib. Also, these colormaps auto-normalize your data.

        :param cmap: a string describing the colormap you want;
            a matplotlib colormap or "blank", which is transparent.
        :param normalize: whether data should be normalized to [0,1].
        :return: a colormap function.
    """

    def blank(x): # transparent
        return np.zeros((np.size(x), 4))

    func = None # the color map
    if cmap == "blank":
        func = blank
    else:
        try:
            func = plt.get_cmap(cmap, 512)
        except:
            func = sns.color_palette(cmap, 512, as_cmap=True)

    def normalize_wrap(x):
        x = x - np.quantile(x, 0.05)

        if np.max(x) < 1e-12: # the values are probably all zero
            return np.full(np.shape(x), 0.5)
        else:
            x = x / np.quantile(x, 0.95)
            x = np.maximum(x, 0)
            x = np.minimum(x, 1) # cutoff
            return x

    if normalize:
        return lambda x: func(normalize_wrap(x))
    else:
        return func
