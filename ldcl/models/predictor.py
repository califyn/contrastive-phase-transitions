import torch
import torch.nn.functional as F
import torch.nn as nn
import os


class TopPredictor(nn.Module):
    def __init__(self, encoder, predictor=None, predictor_output = 3, fine_tuning = False, predictor_hidden = 64):
        super().__init__()

        if predictor:
            self.predictor = predictor
        else:
            predictor_layers = [nn.Linear(3,predictor_hidden),
                              nn.BatchNorm1d(predictor_hidden),
                              nn.ReLU(inplace=True),
                              nn.Linear(predictor_hidden, predictor_hidden),
                              nn.BatchNorm1d(predictor_hidden),
                              nn.ReLU(inplace=True),
                              nn.Linear(predictor_hidden,predictor_output),
                             ]
            self.predictor = nn.Sequential(*predictor_layers)
        
        self.encoder = encoder
        self.net = nn.Sequential(self.predictor,self.encoder)
        
        if not fine_tuning:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.net(x)


# for BYOL and future use
class predictor(nn.Module):
    def __init__(self, in_size = 3, out_size = 3, hidden_size = 64, num_layers = 3, useBatchNorm = False, activation = nn.ReLU(inplace=True), fine_tuning = False):
        super().__init__()
        self.num_layers = num_layers
        self.bn = useBatchNorm
        self.activation = activation

        mlp_layers = [nn.Linear(in_size, hidden_size)]

        for i in range(self.num_layers - 2):
            if self.bn: mlp_layers.append(nn.BatchNorm1d(hidden_size))
            mlp_layers.append(self.activation)
            mlp_layers.append(nn.Linear(hidden_size, hidden_size))

        if self.bn: mlp_layers.append(nn.BatchNorm1d(hidden_size))
        mlp_layers.append(self.activation)
        mlp_layers.append(nn.Linear(hidden_size, out_size))

        self.net = nn.Sequential(*mlp_layers)


    def forward(self, x):
        return self.net(x)

    def save(self, path, name):
        torch.save(self.net, os.path.join(path, f'{name}_predictor.pt'))