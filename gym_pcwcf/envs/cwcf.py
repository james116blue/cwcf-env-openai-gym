import numpy as np

from gym_pcwcf.envs.base import BaseCwcfEnv

class CwcfEnv(BaseCwcfEnv):

    def __init__(self,
                 data_load_fn,
                 lambda_coefficient=1e-3,
                 costs=None,
                 is_binary_classification=False,
                 terminal_reward=None,
                 random_mode=True,
                 **kargs):

        if is_binary_classification and terminal_reward is not None and len(terminal_reward)!=2:
            raise ValueError('Укзана бинарная классификация, но терминальные награды определены не для двух классов')
        terminal_actions_count = 2 if is_binary_classification else None
        terminal_reward = terminal_reward if terminal_reward is not None else None

        super(CwcfEnv, self).__init__(data_load_fn=data_load_fn,
                                      lambda_coefficient=lambda_coefficient,
                                      costs=costs,
                                      terminal_actions_count=terminal_actions_count,
                                      terminal_reward=terminal_reward,
                                      random_mode=random_mode,
                                      **kargs)

        if is_binary_classification:
            self.y[self.y > 1] = 1




