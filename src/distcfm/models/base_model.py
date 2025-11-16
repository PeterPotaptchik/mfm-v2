import torch.nn as nn
from abc import ABC, abstractmethod

def broadcast_to_shape(tensor, shape):
    return tensor.view(-1, *((1,) * (len(shape) - 1)))

class BaseModel(nn.Module, ABC):
    """Abstract base class for all models."""
    def __init__(self):
        super().__init__()

    @abstractmethod
    def v(self, s, t, x, t_cond, x_cond, class_labels=None):
        """Should return the velocity. Must be implemented by subclass."""
        pass

    def X(self, s, t, x, v):
        s = broadcast_to_shape(s, x.shape)
        t = broadcast_to_shape(t, x.shape)
        return x + (t - s) * v

    def X_and_v(self, s, t, x, t_cond, x_cond, class_labels=None):
        v = self.forward(s, t, x, t_cond, x_cond, class_labels=class_labels)
        return self.X(s, t, x, v), v
    
    def forward(self, s, t, x, t_cond, x_cond, class_labels=None):
        """Forward pass that computes the map."""
        v = self.v(s, t, x, t_cond, x_cond, class_labels=class_labels)
        return self.X(s, t, x, v)