# Max-Heinrich Laves
# Institute of Mechatronic Systems
# Leibniz Universität Hannover, Germany
# 2019

import torch

__all__ = ['accuracy', 'Identity', 'kl_loss', 'nentr', 'xavier_normal_init']


def accuracy(input, target):
    _, max_indices = torch.max(input.data, 1)
    acc = (max_indices == target).sum().float() / max_indices.size(0)
    return acc.item()


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def kl_loss(x):
    return -torch.nn.functional.log_softmax(x, dim=1).mean()


def nentr(p, base=None):
    """
    Calculates entropy of p to the base b. If base is None, the natural logarithm is used.
    :param p: batches of class label probability distributions
    :param base: base b
    :return:
    """
    eps = torch.tensor([1e-16], device=p.device)
    if base:
        base = torch.tensor([base], device=p.device, dtype=torch.float32)
        return (p.mul(p.add(eps).log().div(base.log()))).sum(dim=1).abs()
    else:
        return (p.mul(p.add(eps).log())).sum(dim=1).abs()


def xavier_normal_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
