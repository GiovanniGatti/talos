import math
import unittest

import numpy as np
import pytest

from talos import dp


class TestDp(unittest.TestCase):

    def test_time_constant_random_policy_evaluation(self) -> None:
        fare_structure = np.arange(50, 270, 20, dtype=float)

        q = dp.time_constant_random_policy_evaluation(leg_cap=50,
                                                      arrival_rate=70. / 365,
                                                      price_sensitivity=math.log(2) / (2.75 - 1),
                                                      fare_structure=fare_structure,
                                                      horizon=365)

        opt_q = dp.time_constant_dynamic_programming(leg_cap=50,
                                                     arrival_rate=70. / 365,
                                                     price_sensitivity=math.log(2) / (2.75 - 1),
                                                     fare_structure=fare_structure,
                                                     horizon=365)

        assert np.mean(q[-1, -1]) < np.max(opt_q[-1, -1])
        assert 4288.997 == pytest.approx(np.mean(q[-1, -1]), rel=1e-4)


# TODO: could not copy missing tests from original project because these tests have unwanted dependencies

if __name__ == '__main__':
    unittest.main()
