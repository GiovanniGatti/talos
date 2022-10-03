import abc
import random
from typing import Tuple, List

import math


class DemandParameterState:

    def __init__(self, arrival_rate: float, price_sensitivity: float):
        self._arrival_rate = arrival_rate
        self._price_sensitivity = price_sensitivity

    @property
    def arrival_rate(self) -> float:
        return self._arrival_rate

    @arrival_rate.setter
    def arrival_rate(self, _arrival_rate: float) -> None:
        self._arrival_rate = _arrival_rate

    @property
    def price_sensitivity(self) -> float:
        return self._price_sensitivity

    @price_sensitivity.setter
    def price_sensitivity(self, _price_sensitivity: float) -> None:
        self._price_sensitivity = _price_sensitivity


class Sampler(abc.ABC):

    def sample(self) -> Tuple[float, float]:
        pass


class SimpleSampler(Sampler):

    def __init__(self, arrival_rate: float, price_sensitivity: float):
        self._arrival_rate = arrival_rate
        self._price_sensitivity = price_sensitivity

    def sample(self) -> Tuple[float, float]:
        return self._arrival_rate, self._price_sensitivity


class UniformSampler(Sampler):

    def __init__(self, arrival_rate: Tuple[float, float], price_sensitivity: Tuple[float, float]):
        self._arrival_rate = arrival_rate
        self._price_sensitivity = price_sensitivity
        self._rand = random.Random(x=None)

    def sample(self) -> Tuple[float, float]:
        return self._rand.uniform(*self._arrival_rate), self._rand.uniform(*self._price_sensitivity)


class UniformPriceSampler(Sampler):
    """
    Samples the specified price sensitivity range such that the sample values is uniform with respected the policy
    space.

    This class achieves such behavior by sampling uniformly the space of frat5s, which turns out to be linear with
    respect the space of policies.

        .. math::
            d/df[f \cdot \lambda e^{-\gamma(f/f_0-1)}] = 0 \n
            \lambda e^{-\gamma(f/f_0-1)} (1 - f\gamma/f_0) = 0 \n
            f = f_0/\gamma

    :math:`f` in above equations is the price that maximizes the total revenue when no capacity/time constraints are
    present. Thus, if we sample :math:`\gamma` uniformly, the space of policies (here defined by :math:`f`) will follow
    the distribution :math:`1/x`. One way to work around it, is to sample instead in the space of frat5s

        .. math::
            \gamma = \ln(2) / (F_5 - 1) \leftrightarrow \n
            f = f_0 (F_5 - 1) / \ln(2)

    which turns out to be linear with respect the policy space.
    """

    def __init__(self, arrival_rate: Tuple[float, float], price_sensitivity: Tuple[float, float]):
        self._arrival_rate = arrival_rate
        self._price_sensitivity = price_sensitivity
        self._frat5 = (math.log(2) / max(price_sensitivity) + 1, math.log(2) / min(price_sensitivity) + 1)
        self._rand = random.Random(x=None)

    def sample(self) -> Tuple[float, float]:
        opt_frat5 = self._rand.uniform(*self._frat5)
        price_sensitivity = math.log(2) / (opt_frat5 - 1)
        arrival_rate = self._rand.uniform(*self._arrival_rate)
        return arrival_rate, price_sensitivity


class DiscreteSampler(Sampler):

    def __init__(self, arrival_rate: List[float], price_sensitivity: List[float]):
        self._arrival_rate = tuple(arrival_rate)
        self._price_sensitivity = tuple(price_sensitivity)
        self._rand = random.Random(x=None)

    def sample(self) -> Tuple[float, float]:
        return self._rand.choice(self._arrival_rate), self._rand.choice(self._price_sensitivity)
