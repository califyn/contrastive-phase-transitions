import numpy as np
import torch
import math
import tqdm
import torch.nn as nn

from ..tools.device import t2np

MAX_INPUT = 1000

def embed(encoder_location_or_encoder, orbits_dataset, device=None):
    """
        Embed a dataset into its representations, while also neatly packaging conserved
        quantities and inputs.

        :param encoder_location_or_encoder: either a string (path to encoder)
            or already loaded PyTorch encoder
        :param orbits_dataset: a ConservationDataset object loaded from our datasets
        :return: a tuple of (1) a 2D array of encoder outputs (representations)
            and (2) a dictionary of H, L, phi0, x, y, v.x, v.y
    """

    if isinstance(encoder_location_or_encoder, list):
        components = [torch.load(model_component, map_location=torch.device('cpu')) for model_component in encoder_location_or_encoder]
        branch_encoder = nn.Sequential(*components)
    elif isinstance(encoder_location_or_encoder, str):
        branch_encoder = torch.load(encoder_location_or_encoder, map_location=torch.device('cpu'))
    else:
        print("using encoder")
        branch_encoder = encoder_location_or_encoder
    if device is not None:
        branch_encoder.to(device)
    branch_encoder.eval()

    data = orbits_dataset.data
    data = data.reshape([data.shape[0] * data.shape[1]] + list(data.shape)[2:])
    data = torch.from_numpy(data)

    data_slices = [data[MAX_INPUT * i:min(data.shape[0], MAX_INPUT * (i + 1))] for i in range(0, math.floor(data.shape[0] / MAX_INPUT) + 1)]
    if data_slices[-1].shape[0] == 0:
        data_slices = data_slices[:-1]
    encoder_outputs = []
    for dslice in tqdm.tqdm(data_slices):
        encoder_outputs.append(t2np(branch_encoder(dslice.type(torch.float32).to(device))))
    encoder_outputs = np.concatenate(encoder_outputs, axis=0)

    values = {}
    for k, v in orbits_dataset.bundle.items():
        if k == "idxs_":
            continue
        if len(v.shape) == 3:
            values[k] = v[:, :, 0]
        elif len(v.shape) == 2:
            values[k] = v
        else:
            raise NotImplementedError

        if v.shape[1] != encoder_outputs.shape[0] / v.shape[0]: # conserved quantities
            values[k] = np.repeat(values[k], encoder_outputs.shape[0] / v.shape[0], axis=1)

        values[k] = values[k].flatten()

    print(f"First output val: {encoder_outputs.flatten()[0]}")

    return encoder_outputs, values
