import numpy as np
import gym
from gym import  spaces

class BaseCwcfEnv(gym.Env):
    def __init__(self,
                 data_load_fn,
                 lambda_coefficient=0,
                 costs=None,
                 terminal_actions_count=None,
                 terminal_reward = None,
                 random_mode=True,
                 **kargs):

        data = data_load_fn()
        self.data_x, self.data_y = data[:, 0:-1].astype('float32'), data[:, -1].astype('int32')
        self.DATA_LEN = self.data_x.shape[0]  # объем выборки
        self.FEATURE_DIM = kargs['FEATURE_DIM'] if 'FEATURE_DIM' in kargs else self.data_x.shape[1] #количество признаков
        self.TERMINAL_ACTIONS = terminal_actions_count if terminal_actions_count else len(np.unique(self.data_y)) #здесь количество классов
        self.ACTION_DIM = self.FEATURE_DIM + self.TERMINAL_ACTIONS
        self.costs = np.ones(self.FEATURE_DIM) if costs is None else costs
        self.lambda_coefficient = lambda_coefficient  # стоимость добывания значения признака - д.б. положительным
        self.terminal_reward = terminal_reward #нужно реализовывать

        self.mask = np.zeros((self.FEATURE_DIM), dtype=np.float32)  # mask - z vector
        self.x = np.zeros((self.FEATURE_DIM), dtype=np.float32)  # sample features
        self.y = 0

        min_value = kargs['MIN_VALUE'] if 'MIN_VALUE' in kargs else np.finfo(np.float32).min
        max_value = kargs['MAX_VALUE'] if 'MAX_VALUE' in kargs else np.finfo(np.float32).max
        self.action_space = spaces.Discrete(self.ACTION_DIM)
        self.observation_space = spaces.Box(low=min_value, high=max_value,
                                            shape=(2 * self.FEATURE_DIM,), dtype=np.float32)

        self.random_mode = random_mode
        self.idx = -1  # индекс текущего экземпляра, определяющего состояние среды

    def step(self, action):
        done = False

        if (action<self.TERMINAL_ACTIONS):
            r = self.terminal_reward[self.y][action]
            done = True
            s_next = self._get_state(self.x, self.mask)
        else:
            action_f = np.clip(action - self.TERMINAL_ACTIONS, 0, self.FEATURE_DIM) #cast action
            self.mask[action_f] = 1
            r = -self.costs[action_f] * self.lambda_coefficient   #estimate reward for feature action
            s_next = self._get_state(self.x, self.mask)

        return (s_next, r, done, {})

    def reset(self):

        self.mask[:] = 0
        if self.random_mode:
            self.idx = np.random.randint(0, self.DATA_LEN)
        else:  # идем по порядку
            self.idx = (self.idx + 1) % self.DATA_LEN

        self.x, self.y = self._generate_sample()
        s = self._get_state(self.x, self.mask)
        return s

    def action_masks(self):
        return np.concatenate([np.ones(self.TERMINAL_ACTIONS), np.logical_not(self.mask)])

    def render(self, mode='human'):
        pass

    def close(self):
        del self.data_x
        del self.data_y

    def _generate_sample(self):
        x = self.data_x[self.idx]  # sample features
        y = self.data_y[self.idx]  # class
        return (x, y)

    @staticmethod
    def _get_state(x, m):  # генерация наблюдаемого для агента состояния
        x_ = (x * m)  # скрытие неизвестных признаков
        s = np.concatenate((x_, m), axis=0).astype(np.float32)
        return s