import numpy as np
from scipy import signal

class Tapering:
    def __init__(self, alpha=0.3, key='X'):
        self.alpha = alpha  # ضریب تیپرینگ
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key

    def __call__(self, state_dict):
        x, metadata = state_dict[self.key[0]]
        taper = signal.windows.tukey(x.shape[-1], self.alpha)
        x = x * taper
        state_dict[self.key[1]] = (x, metadata)
