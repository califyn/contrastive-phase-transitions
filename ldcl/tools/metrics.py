import torch.nn.functional as F
import torch
import random
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
import math

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# modified from
# https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.\
# ipynb#scrollTo=RI1Y8bSImD7N

# test using a knn monitor
def knn_eval(model, traind, testd, k=200, device=torch.device('cpu'), swapc=False, mp=False):
    if not targets:
        targets = memory_data_loader.dataset[:][2]#.dataset.targets
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in memory_data_loader:
            data = input1.type(torch.float32).to(device)
            if swapc:
                data = torch.swapaxes(input1, 1, 3)

            if mp:
                with autocast():
                    feature = net(data.to(device=device, non_blocking=True))
                feature = feature.to(torch.float)
            else:
                feature = net(data.to(device=device, non_blocking=True))

            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(targets, device=feature_bank.device).long()
        # loop test data to predict the label by weighted knn search
        for data, _, target in test_data_loader:
            data, target = data.to(device=device, non_blocking=True), target.to(device=device, non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, k, t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
    return total_top1 / total_num


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels

def adjust_learning_rate(epochs, warmup_epochs, base_lr, optimizer, loader, step):
    max_steps = epochs * len(loader)
    warmup_steps = warmup_epochs * len(loader)
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = 0
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def lin_eval(encoder, train_loader, test_loader, ind=None, do_print=False, device=None, epochs=100, mp=False, swapc=False):
    classifier = nn.Linear(512, 10).to(device)
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        momentum=0.9,
        lr=30,
        weight_decay=0
    )
    scaler = GradScaler()

    # training
    for e in range(1, epochs + 1):
        # declaring train
        classifier.train()
        encoder.eval()
        # epoch
        for it, (inputs, _, y) in enumerate(train_loader, start=(e - 1) * len(train_loader)):
            # adjust
            adjust_learning_rate(epochs=epochs,
                                 warmup_epochs=0,
                                 base_lr=30,
                                 optimizer=optimizer,
                                 loader=train_loader,
                                 step=it)
            # zero grad
            classifier.zero_grad()

            def forward_step():
                nonlocal y

                with torch.no_grad():
                    b = encoder(inputs.to(device))
                logits = classifier(b)
                y = y.long()
                loss = F.cross_entropy(logits, y.to(device))
                return loss

            # optimization step
            if mp:
                with autocast():
                    loss = forward_step()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = forward_step()
                loss.backward()
                optimizer.step()

        if e % 10 == 0 or e == epochs:
            accs = []
            classifier.eval()
            for idx, (images, _, labels) in enumerate(test_loader):
                with torch.no_grad():
                    if mp:
                        with autocast():
                            b = encoder(images.to(device))
                            preds = classifier(b).argmax(dim=1)
                    else:
                        b = encoder(images.to(device))
                        preds = classifier(b).argmax(dim=1)
                    hits = (preds == labels.to(device)).sum().item()
                    accs.append(hits / b.shape[0])
            accuracy = np.mean(accs)
            # final report of the accuracy
            if do_print:
                line_to_print = (
                    f'seed: {ind} | accuracy (%) @ epoch {e}: {accuracy:.2f}'
                )
                print(line_to_print)
    return accuracy


def eval_on_loader(model, loader, loss, device='cpu'):
    model.eval()

    with torch.no_grad():
        losses = []
        for it, (input1, input2, y) in enumerate(loader):
            input1 = input1[:,0,:].type(torch.float32).to(device)
            input2 = input2[:,0,:].type(torch.float32).to(device)

            z1 = model(input1)
            z2 = model(input2)

            losses.append(loss(z1, z2).cpu())

    return np.mean(np.array(losses))
"""
import torch
import math
import sys

if 'sklearnex' in sys.modules:
    from sklearnex import patch_sklearn
    patch_sklearn()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from .device import t2np

def get_entr(model, traind, testd, device=torch.device('cpu'), prop=1.0, swapc=False):
    model.eval()
    with torch.no_grad():
        train_en = []
        train_tr = []
        for it, (input1, input2, y) in enumerate(traind):
            if it > len(traind) * prop:
                break
            input1 = input1.type(torch.float32).to(device)
            if swapc:
                input1 = torch.swapaxes(input1, 1, 3)
            train_en.append(model.encoder(input1))
            train_tr.append(y)
        train_en = torch.cat(train_en, axis=0)[:math.floor(len(traind.dataset) * prop)]
        train_tr = torch.cat(train_tr)[:math.floor(len(traind.dataset) * prop)]

        test_en = []
        test_tr = []
        for it, (input1, input2, y) in enumerate(testd):
            if it > len(testd) * prop:
                break
            input1 = input1.type(torch.float32).to(device)
            if swapc:
                input1 = torch.swapaxes(input1, 1, 3)
            test_en.append(model.encoder(input1))
            test_tr.append(y)
        test_en = torch.cat(test_en, axis=0)[:math.floor(len(testd.dataset) * prop)]
        test_tr = torch.cat(test_tr)[:math.floor(len(testd.dataset) * prop)]
    return train_en, train_tr, test_en, test_tr

def knn_eval(model, traind, testd, k=3, device=torch.device('cpu'), prop=1.0, swapc=False):
    train_en, train_tr, test_en, test_tr = get_entr(model, traind, testd, device=device, prop=prop, swapc=swapc)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(t2np(train_en), t2np(train_tr))
    return knn.score(t2np(test_en), t2np(test_tr))

def lin_eval(model, traind, testd, device=torch.device('cpu'), prop=1.0, swapc=False):
    train_en, train_tr, test_en, test_tr = get_entr(model, traind, testd, device=device, prop=prop, swapc=swapc)

    train_en = train_en / (1e-10 + torch.linalg.vector_norm(train_en, dim=1, keepdim=True))
    test_en = test_en / (1e-10 + torch.linalg.vector_norm(test_en, dim=1, keepdim=True))
    train_en = train_en - torch.mean(train_en, dim=0, keepdim=True)
    test_en = test_en - torch.mean(test_en, dim=0, keepdim=True)
    train_en = train_en / (1e-10 + torch.std(train_en, (0,), True, keepdim=True))
    test_en = test_en / (1e-10 + torch.std(test_en, (0,), True, keepdim=True))

    lr = LogisticRegression()
    lr.fit(t2np(train_en), t2np(train_tr))
    return lr.score(t2np(test_en), t2np(test_tr))
"""
