import numpy as np

from ldcl.plot.plot import VisPlot
from ldcl.plot.embed import embed
from ldcl.plot.color import get_cmap

from ldcl.data.physics import get_dataset
from ldcl.tools.device import get_device

from sklearn.decomposition import PCA
import argparse

import subprocess

device = get_device(idx=7)


def main_plot(args):
    dataset, _ = get_dataset("data_configs/short_trajectories.json", "./saved_datasets")
    single_orbit, _ = get_dataset("data_configs/single_orbit.json", "./saved_datasets")

    model_encoder_path = f"./saved_models/{args.fname}/{args.id}_encoder.pt"
    # model_projector_path = f"./saved_models/{args.fname}/{args.id}_projector.pt"
    # model_predictor_path = f"./saved_models/{args.fname}/{args.id}_predictor.pt"

    model_path = model_encoder_path

    embeds, vals = embed(model_path, dataset, device=device)
    so_embeds, so_vals = embed(model_path, single_orbit, device=device)

    so_embeds = so_embeds[::10] # make it smaller
    for key in so_vals.keys():
        so_vals[key] = so_vals[key][::10]

    # mask = np.less(vals['phi0'], 3.14) # apply a mask to the data to be visualized
    # embeds = embeds[mask]
    # for key in vals.keys():
    #     vals[key] = vals[key][mask]

    # Colors

    viridis = get_cmap('viridis')
    plasma = get_cmap('plasma')
    blank = get_cmap('blank')

    # Plot

    def cmap_three():
        nonlocal embeds

        plot = VisPlot(3, num_subplots=3) # 3D plot, 2 for 2D plot
        print(embeds.shape)

        # You can use this to plot x, v.x, y and v.y (inputs) as well
        # plot.add_with_cmap(embeds, vals, cmap=["husl", "viridis", "viridis", "viridis", "viridis"], cby=["phi0", "H", "L", "x", "v.x"], size=1.5, outline=False)
        # plot.add_with_cmap(so_embeds, so_vals, cmap=["husl", "viridis", "viridis", "viridis", "viridis"], cby=["phi0", "H", "L", "x", "v.x"], size=2.5, outline=True)

        plot.add_with_cmap(embeds, vals, cmap=["husl", "viridis", "viridis"], cby=["phi0", "H", "L"], size=1.5, outline=False)
        plot.add_with_cmap(so_embeds, so_vals, cmap=["husl", "viridis", "viridis"], cby=["phi0", "H", "L"], size=2.5, outline=True)
        return plot

    def cmap_one(): # display only a single quantity
        plot = VisPlot(3)
        plot.add_with_cmap(embeds, vals, cmap="viridis", cby="L", size=2.5, outline=False)

        return plot

    plot = cmap_three()

    plot.show()
    if args.server:
        subprocess.run('python -m http.server', shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', default='default', type=str)
    parser.add_argument('--image', action='store_true')
    parser.add_argument('--id', default='final', type=str) # id of the model to load
    parser.add_argument('--server', action='store_true') # run a server to view the plot

    args = parser.parse_args()
    main_plot(args)
