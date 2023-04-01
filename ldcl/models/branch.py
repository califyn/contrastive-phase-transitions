import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import os

from torchvision.models import resnet18

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class sslModel(nn.Module):
    def __init__(self, encoder=None, projector=None, predictor=None):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.predictor = predictor

    def forward(self, x):
        out = self.encoder(x)
        if self.projector is not None:
            out = self.projector(out)

        return out

    def save(self, path, name):
        torch.save(self.encoder, os.path.join(path, f'{name}_encoder.pt'))
        if self.projector is not None:
            torch.save(self.projector, os.path.join(path, f'{name}_projector.pt'))
        if self.predictor is not None:
            torch.save(self.predictor, os.path.join(path, f'{name}_predictor.pt'))

    def params(self, lr):
        params_list = list(self.encoder.parameters())
        if self.projector is not None:
            params_list += list(self.projector.parameters())
        if self.predictor is not None:
            params_list += list(self.predictor.parameters())

        return [{
            'name': 'base',
            'params': params_list,
            'lr': lr
        }]

class branchEncoder(nn.Module):
    def __init__(self, encoder_in = 4, encoder_out = 3, encoder_hidden = 64, num_layers = 4, useBatchNorm = False, activation = nn.ReLU(inplace=True)):
        super().__init__()
        self.num_layers = num_layers
        self.bn = useBatchNorm
        self.activation = activation

        encoder_layers = [nn.Linear(encoder_in,encoder_hidden)]

        for i in range(self.num_layers - 2):
            if self.bn: encoder_layers.append(nn.BatchNorm1d(encoder_hidden))
            encoder_layers.append(self.activation)
            encoder_layers.append(nn.Linear(encoder_hidden, encoder_hidden))

        if self.bn: encoder_layers.append(nn.BatchNorm1d(encoder_hidden))
        encoder_layers.append(self.activation)
        encoder_layers.append(nn.Linear(encoder_hidden, encoder_out))

        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        return self.encoder(x)
class branchImageEncoder(nn.Module):
    def __init__(self, encoder_out = 3, encoder_hidden = 64, num_layers = 4, useBatchNorm = False, activation = nn.ReLU(inplace=True)):
        super().__init__()
        self.num_layers = num_layers
        self.bn = useBatchNorm
        self.activation = activation
        self.encoder = resnet18()

        encoder_layers = [nn.Linear(512,encoder_hidden)]

        for i in range(self.num_layers - 2):
            if self.bn: encoder_layers.append(nn.BatchNorm1d(encoder_hidden))
            encoder_layers.append(self.activation)
            encoder_layers.append(nn.Linear(encoder_hidden, encoder_hidden))

        if self.bn: encoder_layers.append(nn.BatchNorm1d(encoder_hidden))
        encoder_layers.append(self.activation)
        encoder_layers.append(nn.Linear(encoder_hidden, encoder_out))

        self.encoder.fc = nn.Sequential(*encoder_layers)

    def forward(self, x):
        x = torch.swapaxes(x, 1, 3)
        return self.encoder(x)

# projectionHead
# implementing the projection head described in simclr paper

class projectionHead(nn.Module):
    def __init__(self, head_in = 3, head_out = 4, hidden_size = 64, num_layers = 3, activation = nn.ReLU(inplace=True), useBatchNorm=False):
        super().__init__()
        self.num_layers = num_layers
        self.activation = activation
        self.bn = useBatchNorm

        layers = [nn.Linear(head_in, hidden_size)]
        for i in range(self.num_layers - 2):
            if self.bn: 
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(self.activation)
            layers.append(nn.Linear(hidden_size, hidden_size))
        if self.bn:
            layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(self.activation)
        layers.append(nn.Linear(hidden_size, head_out))

        self.head = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.head(x)

class predictor(nn.Module):
    def __init__(self, size, hidden_size = 64, useBatchNorm=False, activation=nn.ReLU(inplace=True), num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.activation = activation
        self.bn = useBatchNorm

        layers = [nn.Linear(size, hidden_size)]
        for i in range(self.num_layers - 2):
            if self.bn: 
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(self.activation)
            layers.append(nn.Linear(hidden_size, hidden_size))
        if self.bn:
            layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(self.activation)
        layers.append(nn.Linear(hidden_size, size))

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
