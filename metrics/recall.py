import torch

from metrics.base_metric import BaseMetric


class Recall(BaseMetric):
    def __init__(self, name='Recall'):
        super().__init__(name)
        self._tp = 0
        self._fn = 0

    def reset(self):
        super().reset()
        self._tp = 0
        self._fn = 0

    def update(self, input, target):
        y_pred = torch.round(torch.sigmoid(input))
        self._tp += (y_pred * target).sum()
        self._fn += ((1 - y_pred) * target).sum()
        self._num_samples += target.shape[0]

    def compute(self):
        recall = self._tp / (self._tp + self._fn)
        return recall
