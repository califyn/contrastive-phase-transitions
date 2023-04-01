# Phase Transitions in Contrastive Learning

This is the repository for _Studying Phase Transitions in Contrastive Learning with Physics-Inspired Datasets_ to appear in the Physics for Machine Learning Workshop at ICLR 2023.

First, to set up all the requirements, run the following from the root:
```shell
pip install -e .
pip install requirements.txt
```

## Training loop

To train a model on the Kepler dataset with a given alpha, run from the root:

```shell
python main.py
```

Training takes approximately 5-10 minutes on a consumer desktop.

## Visualization

After training a model, run from the root:

```shell
python visualize.py --id={id}
```

where put ``id`` as ``start`` for the newly-initialized model, ``final`` for the final model, and the epoch number (by default every 20 epochs are saved, so this only works if you select multiples of 20). This will open up the visualization as HTML on your desktop (see options for other ways to display the visualization).

## Recreating the phase transition plot

To recreate Figure ?? in the paper, run the following:

```shell
python main_for_transition.py
python transition_visualize.py
```
