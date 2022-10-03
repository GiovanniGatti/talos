import abc
from typing import Tuple, Optional, Type, Any, Dict

import gym
import numpy as np


class _CircularQueue:

    def __init__(self, shape: Tuple[int, ...], size: int, dtype: Optional[Type] = float):
        self._queue = np.empty(shape=(size,) + shape, dtype=dtype)
        self._queue[:] = np.nan
        self._size = size
        self._tail = 0

    def enqueue(self, new_data: np.ndarray) -> None:
        self._queue[self._tail] = new_data
        self._tail = (self._tail + 1) % self._size

    def as_ndarray(self) -> np.ndarray:
        x = np.where(~np.isnan(self._queue), self._queue, 0)
        block1 = np.arange(0, self._tail)[::-1]
        block2 = np.arange(self._tail, self._size)[::-1]
        order = np.concatenate((block1, block2))
        return x[order]


class History(abc.ABC):

    @abc.abstractmethod
    def store(self, data: np.ndarray) -> None:
        pass

    @abc.abstractmethod
    def distribution(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def raw(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def nbins(self) -> int:
        pass


class SumHistory(History):

    def __init__(self, nbins: int, horizon: int, dtype=float):
        self._hist = _CircularQueue(shape=(nbins,), size=horizon, dtype=dtype)
        self._nbins = nbins

    def store(self, data: np.ndarray) -> None:
        self._hist.enqueue(data)

    def distribution(self) -> np.ndarray:
        return np.sum(self._hist.as_ndarray(), axis=0)

    def raw(self) -> np.ndarray:
        return self._hist.as_ndarray()

    def nbins(self) -> int:
        return self._nbins


class BookingHistoryWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, history: History):
        super().__init__(env)
        self._history = history
        self._n = self._history.nbins()

    def step(self, action: Any) -> Any:
        o, r, done, infos = super().step(action)
        self._store_bookings(action, infos)
        return o, r, done, infos

    def _store_bookings(self, action: Any, infos: Dict[Any, Any]) -> None:
        daily_bkgs = infos['daily_bkgs']
        action = action[daily_bkgs > 0]
        daily_bkgs = daily_bkgs[daily_bkgs > 0]
        all_actions = np.broadcast_to(np.arange(self._n), (action.shape[0], self._n)).T
        idx = all_actions == action
        zeros = np.zeros(action.shape[0])
        bookings = np.sum(np.where(idx, daily_bkgs, zeros), axis=-1)
        self._history.store(bookings)

    def reset(self) -> Any:
        org_step = self.unwrapped.step

        def spy_step(action: Any):
            o, r, done, infos = org_step(action)
            self._store_bookings(action, infos)
            return o, r, done, infos

        self.unwrapped.step = spy_step
        obs = super().reset()
        self.unwrapped.step = org_step
        return obs


class OfferHistoryWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, history: History):
        super().__init__(env)
        self._history = history

    def step(self, action: Any) -> Any:
        o, r, done, infos = super().step(action)
        self._store_offers(action, infos)
        return o, r, done, infos

    def _store_offers(self, action: Any, infos: Dict[Any, Any]) -> None:
        mask = infos['incomplete_flights']
        idx, counts = np.unique(action[mask], return_counts=True)
        offers = np.zeros(self._history.nbins())
        offers[idx] = counts
        self._history.store(offers)

    def reset(self) -> Any:
        org_step = self.unwrapped.step

        def spy_step(action: Any):
            o, r, done, infos = org_step(action)
            self._store_offers(action, infos)
            return o, r, done, infos

        self.unwrapped.step = spy_step
        obs = super().reset()
        self.unwrapped.step = org_step
        return obs
