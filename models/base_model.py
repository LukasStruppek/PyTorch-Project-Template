import torch
import torch.nn as nn
from abc import abstractmethod
import numpy as np

class BaseModel(nn.Module):
    """
    Base model for all PyTorch models.
    """

    def __init__(self, name=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0') if self.use_cuda else torch.device('cpu')
    
    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def test(self):
        pass

    def set_parameter_requires_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad = requires_grad

    def __str__(self):
        num_params = np.sum([param.numel() for param in self.parameters()])
        if self.name:
            return self.name + '\n' + super().__str__() + f'\n Total number of parameters: {num_params}'
        else:
            return super().__str__() + f'\n Total number of parameters: {num_params}'