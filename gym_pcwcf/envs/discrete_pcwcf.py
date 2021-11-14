import numpy as np

from gym_pcwcf.envs.base import BaseCwcfEnv

class DiscretePcwcfEnv(BaseCwcfEnv):
    metadata = {'render.modes:': ['human']}
    
    def __init__(self,
                 data_load_fn,
                 lambda_coefficient=1e-3,
                 costs=None,
                 probability_discretization=11, #дискретизация вида array([1. , 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0. ])
                 random_mode=True,
                 eps=1e-15,
                 **kargs):

        #вероятности вида (для probability_discretization=11) [1-epsilon , 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, epsilon ]
        terminal_reward = np.log(np.clip(np.stack([np.linspace(1, 0, probability_discretization),
                                                   np.linspace(0, 1, probability_discretization)]),
                                         eps,
                                         1 - eps))

        super(DiscretePcwcfEnv, self).__init__(data_load_fn=data_load_fn,
                                               lambda_coefficient=lambda_coefficient,
                                               costs=costs,
                                               terminal_actions_count=probability_discretization,
                                               terminal_reward=terminal_reward,
                                               random_mode=random_mode,
                                               **kargs)

if __name__=="__main__":
    import numpy as np
    import gym
    import os
    # ============================================DATASET
    from gym_pcwcf.config.config_synthetic_simple import DATASET_SETTING, DATA_LOAD_FN, MODEL_PATH
    FEATURE_DIM = DATASET_SETTING['FEATURE_DIM']
    TERMINAL_ACTIONS = 11
    ACTION_DIM = FEATURE_DIM + TERMINAL_ACTIONS
    TRAIN_DATA_LEN = DATASET_SETTING['TRAIN_DATA_LEN']
    GYM_VERSION_NAME = 'gym_pcwcf:pcwcf-v0'
    # ============================================/DATASET
    env = gym.make(GYM_VERSION_NAME,
                   data_load_fn=DATA_LOAD_FN,
                   lambda_coefficient=1e-3,
                   costs=None,
                   discretization_class_number=11,
                   random_mode=True,
                   eps=1e-15)
    env.reset()
    x, y = env.x, env.y

    target_rewards = np.ones(ACTION_DIM - TERMINAL_ACTIONS + 1) * -0.01
    true_rewards = np.array( [env.step(action)[1] for action in range(ACTION_DIM-1, TERMINAL_ACTIONS-1, -1)] )
    terminal_action = 8
    target_rewards[-1] = np.log(0.8)
    terminal_reward = env.step(terminal_action)[1]
    true_rewards = np.append(true_rewards, terminal_reward)
    assert np.testing.assert_array_equal(true_rewards, target_rewards)