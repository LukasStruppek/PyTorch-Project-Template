import torch

from metrics.base_metric import BaseMetric


class Accuracy(BaseMetric):
    def __init__(self, name='Accuracy@1'):
        super().__init__(name)
        self._num_corrects = 0

    def reset(self):
        super().reset()
        self._num_corrects = 0

    def update(self, input, target):
        y_pred = torch.argmax(input, dim=1)
        self._num_corrects += torch.sum(y_pred == target).item()
        self._num_samples += target.shape[0]

    def compute(self):
        accuracy = self._num_corrects / self._num_samples
        return accuracy


class AccuracyTopK(BaseMetric):
    def __init__(self, name='Accuracy@', k=5):
        super().__init__(name + str(k))
        self.k = k
        self._num_corrects = 0

    def reset(self):
        super().reset()
        self._num_corrects = 0

    def update(self, input, target):
        y_pred = torch.topk(input, dim=1, k=self.k).indices
        num_corrects = 0
        for k in range(self.k):
            num_corrects += torch.sum(y_pred[:, k] == target).item()
        self._num_corrects += num_corrects
        self._num_samples += target.shape[0]

    def compute(self):
        accuracy = self._num_corrects / self._num_samples
        return accuracy
