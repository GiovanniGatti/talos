import unittest

import numpy as np
from ray.rllib.policy import sample_batch

from talos.continuing import postprocess


class TestAvgRwdFn(unittest.TestCase):

    def test_avg_cost_subtracts_each_reward_observation_by_the_mean_of_the_samples(self) -> None:
        rollout = sample_batch.SampleBatch({sample_batch.SampleBatch.REWARDS: np.array([0., 1., 2.])})
        any_vpred_t = np.random.random(rollout[sample_batch.SampleBatch.REWARDS].shape[0])

        postprocess.avg_cost_fn(rollout, any_vpred_t)

        assert (rollout[sample_batch.SampleBatch.REWARDS] == np.array([-1., 0., 1.])).all()

    def test_diff_td_subtracts_each_reward_observation_by_the_mean_of_td_errors(self) -> None:
        rollout = sample_batch.SampleBatch({sample_batch.SampleBatch.REWARDS: np.array([0., 1., 2.])})
        vpred_t = np.array([-2., 0., -1., 1.])

        postprocess.diff_td_fn(rollout, vpred_t)

        # Mean delta_t
        # r     =  0,  1,  2
        # +
        # v_s'  =  0, -1,  1
        # -
        # v_s   = -2,  0, -1
        # =
        # delta =  2,  0,  4
        # mean(delta) = (2 + 0 + 4) / 3 = 2
        assert (rollout[sample_batch.SampleBatch.REWARDS] == np.array([-2., -1., 0.])).all()

    def test_rvi_subtracts_each_reward_observation_by_the_first_predicted_value(self) -> None:
        rollout = sample_batch.SampleBatch({sample_batch.SampleBatch.REWARDS: np.array([0., 1., 2.])})
        vpred_t = np.array([-2., 0., -1., 1.])

        postprocess.rvi_fn(rollout, vpred_t)

        assert (rollout[sample_batch.SampleBatch.REWARDS] == np.array([2., 3., 4.])).all()


if __name__ == '__main__':
    unittest.main()
