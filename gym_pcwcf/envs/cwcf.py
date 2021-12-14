import numpy as np

from gym_pcwcf.envs.base import BaseCwcfEnv

class CwcfEnv(BaseCwcfEnv):

    def __init__(self,
                 dataset_name='cicids2017_webattacks',
                 mode='TRAIN',
                 lambda_coefficient=0, #должны быть положительными - умножаются потом на -1
                 costs=None,
                 is_binary_classification=True,
                 terminal_reward=None,
                 random_mode=True,
                 **kargs):

        if is_binary_classification and terminal_reward is not None and len(terminal_reward)!=2:
            raise ValueError('Укзана бинарная классификация, но терминальные награды определены не для двух классов')
        terminal_actions_count = 2 if is_binary_classification else None
        terminal_reward = terminal_reward if terminal_reward is not None else None

        super(CwcfEnv, self).__init__(dataset_name=dataset_name,
                                      mode=mode,
                                      lambda_coefficient=lambda_coefficient,
                                      costs=costs,
                                      terminal_actions_count=terminal_actions_count,
                                      terminal_reward=terminal_reward,
                                      random_mode=random_mode,
                                      **kargs)

        if is_binary_classification:
            self.data_y[self.data_y > 1] = 1

    def step(self, action):
        s_next, r, done, info = super(CwcfEnv, self).step(action)
        if (action<self.TERMINAL_ACTIONS):
            info['y_true'] = self.y
            info['y_predict'] = action

        return (s_next, r, done, info)
