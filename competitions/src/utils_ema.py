# === Added for exp006 improvement ===
import copy
import torch
import torch.nn as nn

class ModelEMA:
    """
    Model Exponential Moving Average.
    Maintains a moving average of model parameters for better validation stability.
    """
    def __init__(self, model, decay=0.999, device=None):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # Perform EMA on different device from model if set
        if self.device is not None:
            self.module.to(device=self.device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
        
    def to(self, device):
        self.module.to(device=device)
        self.device = device
        return self
