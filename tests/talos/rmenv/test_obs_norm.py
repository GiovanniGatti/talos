import random
import unittest
from typing import Any

import gym
import numpy as np
import pytest
from gym import spaces

from talos import rmenv


def any_action_space() -> spaces.Space:
    if random.random() > 0.5:
        return spaces.Box(low=random.randint(-5, -1),
                          high=random.randint(1, 5),
                          shape=(random.randint(1, 3),),
                          dtype=float)
    return spaces.MultiDiscrete((random.randint(1, 5), random.randint(2, 4)))


class _DummyEnv(gym.Env):

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def step(self, action: Any) -> Any:
        assert self.action_space.contains(action)
        return self.observation_space.sample()

    def reset(self) -> Any:
        return self.observation_space.sample()


class TestMinMax(unittest.TestCase):

    def test_observation_is_minmax_normalized(self) -> None:
        observation_space = spaces.MultiDiscrete((5, 3, 6, 9))
        action_space = any_action_space()
        env = _DummyEnv(observation_space, action_space)
        wrapped = rmenv.MinMax(env)

        obs = wrapped.observation(np.array([1, 1, 5, 0]))

        assert (obs == np.array([-.5, 0., 1., -1.])).all()

    def test_only_multidiscrete_spaces_are_supported(self) -> None:
        observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=float)
        action_space = any_action_space()
        env = _DummyEnv(observation_space, action_space)

        with pytest.raises(ValueError) as info:
            rmenv.MinMax(env)

        assert str(info.value) == \
               'Only MultiDiscrete spaces are currently supported, but found Box([0.], [1.], (1,), float64)'


class TestOneHot(unittest.TestCase):

    def test_observation_is_onehot_encoded(self) -> None:
        observation_space = spaces.MultiDiscrete(((5, 4), (5, 4)))
        action_space = any_action_space()
        env = _DummyEnv(observation_space, action_space)
        wrapped = rmenv.OneHot(env)

        obs = wrapped.observation(np.array([[1, 2], [4, 0]]))

        assert (obs == np.array([[0, 1, 0, 0, 0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 1, 1, 0, 0, 0]])).all()

    def test_only_observations_with_2_dimensions_are_supported(self) -> None:
        observation_space = spaces.MultiDiscrete((5, 4))
        action_space = any_action_space()
        env = _DummyEnv(observation_space, action_space)

        with pytest.raises(ValueError) as info:
            rmenv.OneHot(env)

        assert str(info.value) == \
               'Unsupported MultiDiscrete spec. Expected 2 dim obs. space, but found [5 4]'

    def test_only_multidiscrete_spaces_are_supported(self) -> None:
        observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=float)
        action_space = any_action_space()
        env = _DummyEnv(observation_space, action_space)

        with pytest.raises(ValueError) as info:
            rmenv.OneHot(env)

        assert str(info.value) == \
               'Expected state space of type MultiDiscrete, but found Box([0.], [1.], (1,), float64)'


if __name__ == '__main__':
    unittest.main()
