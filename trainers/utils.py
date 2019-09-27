import os

import numpy as np

import torch
import torch.nn.functional as F





class Averager(object):
    """The class to calculate the average."""

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits: torch.Tensor,
              label: torch.Tensor) -> float:
    """The function to calculate the .
    Args:
      logits: input logits.
      label: ground truth labels.
    Return:
      The output accuracy.
    """
    pred = F.softmax(logits, dim=1).argmax(dim=1)
    return (pred == label).type(torch.FloatTensor).mean().item()


def compute_confidence_interval(data):
    """The function to calculate the .
    Args:
      data: input records
    Return:
      m: mean value
      pm: confidence interval.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm
