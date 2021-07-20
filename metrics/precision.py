import torch

from metrics.base_metric import BaseMetric


class Precision(BaseMetric):
    def __init__(self, name='Precision'):
        super().__init__(name)
        self._tp = 0
        self._fp = 0

    def reset(self):
        super().reset()
        self._tp = 0
        self._fp = 0

    def update(self, input, target):
        y_pred = torch.round(torch.sigmoid(input))
        self._tp += (y_pred * target).sum()
        self._fp += (y_pred * (1 - target)).sum()
        self._num_samples += target.shape[0]

    def compute(self):
        precision = self._tp / (self._tp + self._fp)
        return precision
