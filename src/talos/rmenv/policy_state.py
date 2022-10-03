import abc
from typing import Tuple

import numpy as np


class PolicyState(abc.ABC):

    @abc.abstractmethod
    def horizon(self) -> int:
        pass

    @abc.abstractmethod
    def initial_cap(self) -> int:
        pass

    @abc.abstractmethod
    def v(self, dtd: int, cap: int) -> float:
        pass

    @abc.abstractmethod
    def pi(self, dtd: int, cap: int) -> float:
        pass

    @abc.abstractmethod
    def __getitem__(self, item: Tuple[np.ndarray, ...]) -> np.ndarray:
        pass

    @abc.abstractmethod
    def update(self, new_arrival_rate: float, new_price_sensitivity: float) -> None:
        pass
