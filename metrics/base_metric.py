from abc import abstractmethod

import torch


class BaseMetric():
    def __init__(self, name):
        self._num_samples = 0
        self.name = name
        super().__init__()

    @abstractmethod
    def reset(self):
        self._num_samples = 0

    @abstractmethod
    def update(self, input, target):
        pass

    @abstractmethod
    def compute(self):
        pass
