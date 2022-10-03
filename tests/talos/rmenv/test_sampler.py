import math
import unittest

import numpy as np
import pytest

from talos import dp
from talos import rmenv


class TestUniformSampler(unittest.TestCase):

    def test_us_samples_uniformly(self) -> None:
        _sampler = rmenv.UniformSampler(arrival_rate=(.3, .3), price_sensitivity=(.4, .6))

        sampled_arrival_rates = []
        sampled_price_sensitivity = []
        for _ in range(5000):
            arrival_rate, price_sensitivity = _sampler.sample()
            sampled_arrival_rates.append(arrival_rate)
            sampled_price_sensitivity.append(price_sensitivity)

        assert all(.3 == x for x in sampled_arrival_rates)

        sampled_price_sensitivity = np.array(sampled_price_sensitivity)
        assert np.mean(sampled_price_sensitivity > .5) == pytest.approx(0.5, abs=0.02)
        assert all(.4 <= x <= .6 for x in sampled_price_sensitivity)


class TestDiscreteSampler(unittest.TestCase):

    def test_dus_samples_uniformly(self) -> None:
        _sampler = rmenv.DiscreteSampler(arrival_rate=[.1], price_sensitivity=[.2, .3])

        sampled_arrival_rates = []
        sampled_price_sensitivity = []
        for _ in range(5000):
            arrival_rate, price_sensitivity = _sampler.sample()
            sampled_arrival_rates.append(arrival_rate)
            sampled_price_sensitivity.append(price_sensitivity)

        assert all(.1 == x for x in sampled_arrival_rates)

        assert sampled_price_sensitivity.count(.2) / len(sampled_price_sensitivity) == pytest.approx(0.5, abs=0.02)
        assert sampled_price_sensitivity.count(.3) / len(sampled_price_sensitivity) == pytest.approx(0.5, abs=0.02)
        assert all(x in [.2, .3] for x in sampled_price_sensitivity)


class TestUniformPriceSampler(unittest.TestCase):

    def test_us_samples_uniformly_on_the_policy_domain(self) -> None:
        _sampler = rmenv.UniformPriceSampler(arrival_rate=(4 / 22, 4 / 22), price_sensitivity=(math.log(2) / (4.3 - 1),
                                                                                               math.log(2) / (1.5 - 1)))

        sampled_arrival_rates = []
        sampled_price_sensitivity = []
        for _ in range(5000):
            arrival_rate, price_sensitivity = _sampler.sample()
            sampled_arrival_rates.append(arrival_rate)
            sampled_price_sensitivity.append(price_sensitivity)

        assert all(4 / 22 == x for x in sampled_arrival_rates)

        collected = []
        for ps in sampled_price_sensitivity:
            policy = dp.time_constant_dynamic_programming(leg_cap=6, arrival_rate=0.01, price_sensitivity=ps,
                                                          fare_structure=np.arange(50, 250, 20),
                                                          horizon=22).argmax(axis=-1)
            distribution = dp.fare_class_distribution(leg_cap=6, arrival_rate=0.01, price_sensitivity=ps,
                                                      fare_structure=np.arange(50, 250, 20), horizon=22, policy=policy)
            collected.append(distribution)

        assert np.mean(collected, axis=0) == pytest.approx(np.repeat(0.1, 10), abs=0.02)


if __name__ == '__main__':
    unittest.main()
