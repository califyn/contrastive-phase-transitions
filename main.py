import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import os
import shutil
import json
import argparse

# ldcl imports
from ldcl.models import branch, predictor
from ldcl.optimizers.lr_scheduler import LR_Scheduler, get_lr_scheduler
from ldcl.data import physics
from ldcl.losses.nce import rmseNCE
from ldcl.plot.plot import plot_loss
from ldcl.tools import metrics
from ldcl.tools.device import get_device, t2np
from ldcl.tools.seed import set_deterministic

import tqdm

device = get_device()

def training_loop(args):
    """
    drop_last: drop the last non_full batch (potentially useful for training weighting etc.)
    pin_memory: speed dataloader transfer to cuda
    num_workers: multiprocess data loading
    """

    # Set alpha and get dataset
    data_config_file = "data_configs/" + args.data_config + ".json"
    with open(data_config_file, "r") as f:
        data_config_wr = json.load(f)

    if args.alpha != 1.0:
        args.data_config = args.data_config + "_temp"

    data_config_file = "data_configs/" + args.data_config + ".json"

    if args.alpha != 1.0:
        data_config_wr["orbit_settings"]["traj_range"] = args.alpha
        with open(data_config_file, "w") as f:
            json.dump(data_config_wr, f)
        
    train_orbits_dataset, folder = physics.get_dataset(data_config_file, "./saved_datasets")
    print(f"Using dataset {folder}...")
    
    # Set up saving files
    save_progress_path = os.path.join("./saved_models", args.fname)
    while os.path.exists(save_progress_path):
        to_del = input("Saved directory already exists. If you continue, you may erase previous training data. Press Ctrl+C to stop now. Otherwise, type 'yes' to continue:")
        if to_del == "yes":
            shutil.rmtree(save_progress_path)
    os.mkdir(save_progress_path)
    shutil.copy(data_config_file, os.path.join(save_progress_path, "data_config.json"))
    train_orbits_loader = torch.utils.data.DataLoader(
        dataset = train_orbits_dataset,
        sampler=torch.utils.data.BatchSampler(
                torch.utils.data.RandomSampler(train_orbits_dataset), batch_size=args.bsz, drop_last=True
            ),
    )

    # Set up many loss experiment
    if args.many_losses:
        val_data_loaders = {
            "05": torch.utils.data.DataLoader(physics.get_dataset("data_configs/short_traj_05.json", "./saved_datasets")[0], drop_last=True, batch_size=args.bsz),
            "10": torch.utils.data.DataLoader(physics.get_dataset("data_configs/short_traj_10.json", "./saved_datasets")[0], drop_last=True, batch_size=args.bsz),
            "20": torch.utils.data.DataLoader(physics.get_dataset("data_configs/short_traj_20.json", "./saved_datasets")[0], drop_last=True, batch_size=args.bsz),
            "50": torch.utils.data.DataLoader(physics.get_dataset("data_configs/short_traj_50.json", "./saved_datasets")[0], drop_last=True, batch_size=args.bsz),
            "100": torch.utils.data.DataLoader(physics.get_dataset("data_configs/orbit_config_default.json", "./saved_datasets")[0], drop_last=True, batch_size=args.bsz)
        }

    # Create model, training utilities, etc.
    encoder = branch.branchEncoder(encoder_out=3, activation=nn.ReLU(), useBatchNorm=True)
    model = branch.sslModel(encoder=encoder)
    model.to(device)
    model.save(save_progress_path, 'start')

    optimizer = torch.optim.SGD(model.params(args.lr), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    lr_scheduler = get_lr_scheduler(args, optimizer, train_orbits_loader)

    def apply_loss(z1, z2, loss_func = rmseNCE):
        loss = 0.5 * loss_func(z1, z2) + 0.5 * loss_func(z2, z1)
        return loss

    # Training metrics
    direct_loss = lambda x, y: apply_loss(x, y, rmseNCE)
    if args.many_losses:
        emtrs = {
            "five": lambda: metrics.eval_on_loader(model, val_data_loaders["05"], direct_loss, device=device),
            "ten": lambda: metrics.eval_on_loader(model, val_data_loaders["10"], direct_loss, device=device),
            "twenty": lambda: metrics.eval_on_loader(model, val_data_loaders["20"], direct_loss, device=device),
            "fifty": lambda: metrics.eval_on_loader(model, val_data_loaders["50"], direct_loss, device=device),
            "full": lambda: metrics.eval_on_loader(model, val_data_loaders["100"], direct_loss, device=device),
        } # training metrics
    else:
        emtrs = {}

    losses = []
    mtrd = {"loss": None, "avg_loss": None} # dictionary of metric values
    saved_metrics = {}

    def update_metrics(t, new_loss=None, losses=None, do_eval=False): # update metric values
        if new_loss is not None:
            mtrd["loss"] = new_loss
        if losses is not None:
            mtrd["avg_loss"] = np.mean(np.array(losses[max(-1 * len(losses), -50):]))

        if do_eval:
            for name, metric in emtrs.items():
                new_val = metric()
                if name in saved_metrics.keys():
                    saved_metrics[name].append(new_val)
                else:
                    saved_metrics[name] = [new_val]
                mtrd[name] = new_val

        t.set_postfix(**mtrd)

    # Training loop
    with tqdm.trange(args.epochs * len(train_orbits_loader)) as t:
        update_metrics(t, do_eval=True)
        for e in range(args.epochs):
            model.train()

            for it, (input1, input2, y) in enumerate(train_orbits_loader):
                model.zero_grad()

                # forward pass
                input1 = input1[0].type(torch.float32).to(device)
                input2 = input2[0].type(torch.float32).to(device)

                z1 = model(input1)
                z2 = model(input2)

                loss = apply_loss(z1, z2, rmseNCE)

                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                # Update metrics
                losses.append(t2np(loss).flatten()[0])

                if e % args.save_every and it == 0:
                    model.save(save_progress_path, f'{e:02d}')
                update_metrics(t, new_loss=loss.item(), losses=losses)
                t.update()

            if e % args.eval_every == args.eval_every - 1:
                update_metrics(t, do_eval=True)

    # Save training logs
    model.save(save_progress_path, 'final')
    losses = np.array(losses)
    np.save(os.path.join(save_progress_path, "loss.npy"), losses)

    for name, slist in saved_metrics.items():
        np.save(os.path.join(save_progress_path, f"{name}.npy"), slist)

    plot_loss(losses, title = args.fname, save_progress_path = save_progress_path)

    return encoder

if __name__ == '__main__':
    pass
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', default='default' , type=str)
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--data_config', default='short_trajectories' , type=str)

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=0.02, type=float)
    parser.add_argument('--bsz', default=512, type=int)
    parser.add_argument('--wd', default=0.001, type=float)
    parser.add_argument('--warmup_epochs', default=5, type=int)

    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--many_losses', action='store_true')

    parser.add_argument('--save_every', default=20, type=int)
    parser.add_argument('--eval_every', default=3, type=int)

    args = parser.parse_args()
    device = get_device(idx=args.device)
    print(f"Training on device: {args.device}")
    training_loop(args)
