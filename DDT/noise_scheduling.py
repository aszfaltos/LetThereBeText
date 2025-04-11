from abc import ABC, abstractmethod
import torch


class NoiseSchedule(ABC):
    @property
    @abstractmethod
    def alpha_bar(self, t):
        raise NotImplementedError

    def sqrt_alpha_bar(self, t):
        raise NotImplementedError

    def sqrt_one_minus_alpha_bar(self, t):
        raise NotImplementedError
    

class SqrtNoiseSchedule(NoiseSchedule):
    def __init__(self, T: int, s: float = 1e-3):
        super().__init__()
        self.T = T
        self._alpha_bar = (
            1 - torch.sqrt(torch.arange(0, T) / T + s)
            )
        self._sqrt_alpha_bar = torch.sqrt(self._alpha_bar)
        self._sqrt_one_minus_alpha_bar = torch.sqrt(1 - self._alpha_bar)
    
    def alpha_bar(self, t):
        return self._alpha_bar[t]
    
    def sqrt_alpha_bar(self, t):
        return self._sqrt_alpha_bar[t]
    
    def sqrt_one_minus_alpha_bar(self, t):
        return self._sqrt_one_minus_alpha_bar[t]
