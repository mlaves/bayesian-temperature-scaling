import torch
import torch.nn as nn
from torch.nn import Identity
import torch.nn.functional as F
from torchvision import models


class BayesianNet(torch.nn.Module):
    def __init__(self, num_classes, model='resnet101', pretrained=False):
        super().__init__()

        assert model in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet169']

        if model == 'resnet18':
            self._model = models.resnet18(pretrained=pretrained)
            fc_in_features = 512
        elif model == 'resnet34':
            self._model = models.resnet34(pretrained=pretrained)
            fc_in_features = 512
        elif model == 'resnet50':
            self._model = models.resnet50(pretrained=pretrained)
            fc_in_features = 2048
        elif model == 'resnet101':
            self._model = models.resnet101(pretrained=pretrained)
            fc_in_features = 2048
        elif model == 'resnet152':
            self._model = models.resnet152(pretrained=pretrained)
            fc_in_features = 2048
        elif model == 'densenet121':
            self._model = models.densenet121(pretrained=pretrained, drop_rate=0)
            fc_in_features = 1024
        elif model =='densenet169':
            self._model = models.densenet169(pretrained=pretrained, drop_rate=0)
            fc_in_features = 1664
        else:
            assert False

        if 'resnet' in model:
            self._model.fc = Identity()
        elif 'densenet' in model:  # densenet
            self._model.classifier = self._model.fc = Identity()

        self._fc = torch.nn.Linear(in_features=fc_in_features, out_features=num_classes)

        self.T = nn.Parameter(torch.tensor(1.0))
        self.N = 25
        self.p = 0.5

    def mc_dropout(self, x):
        x_list = F.dropout(x, p=self.p, training=True)
        x_list = self._fc(x_list).unsqueeze(0)
        for i in range(self.N - 1):
            x_tmp = F.dropout(x, p=self.p, training=True)
            x_tmp = self._fc(x_tmp).unsqueeze(0)
            x_list = torch.cat([x_list, x_tmp], dim=0)
        return x_list

    def forward(self, x, temp_scale=False, bayesian=False):
        if not temp_scale:
            x = self._model(x)
            if not bayesian:
                x = F.dropout(x, p=self.p, training=self.training)
                logits = self._fc(x)
                return logits
            else:
                return self.mc_dropout(x)
        else:
            with torch.no_grad():
                x = self._model(x)
            if not bayesian:
                with torch.no_grad():
                    x = F.dropout(x, p=self.p, training=False)
                    x = self._fc(x)
                return x / F.relu(self.T)
            else:
                with torch.no_grad():
                    mc_mean = self.mc_dropout(x)
                return mc_mean / F.relu(self.T)  # perform temp scaling
