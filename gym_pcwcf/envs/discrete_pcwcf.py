import numpy as np
import gym
from gym import error, spaces, utils

class DiscretePcwcfEnv(gym.Env):
    metadata = {'render.modes:': ['human']}
    
    def __init__(self, data_load_fn,
                 lambda_coefficient=1e-3,
                 costs=None,
                 discretization_class_number=11, #дискретизация вида array([1. , 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0. ])
                 random_mode=True,
                 eps=1e-15,
                 **kargs):
       
        data = data_load_fn()
        self.data_x, self.data_y = data[:, 0:-1].astype('float32'), data[:,-1].astype('int32')
        self.DATA_LEN        = self.data_x.shape[0] #объем выборки
        self.FEATURE_DIM     = kargs['FEATURE_DIM'] if 'FEATURE_DIM' in kargs else self.data_x.shape[1]
        self.TERMINAL_ACTIONS= discretization_class_number
        self.ACTION_DIM      =  self.FEATURE_DIM + self.TERMINAL_ACTIONS
        
        self.costs = np.ones(self.FEATURE_DIM) if costs is None else costs
        self.lambda_coefficient = lambda_coefficient # стоимость добывания значения признака
        self.terminal_reward = np.log(np.clip(np.stack([np.linspace(1, 0, self.TERMINAL_ACTIONS),
                                                        np.linspace(0, 1, self.TERMINAL_ACTIONS)]),
                                              eps,
                                              1 - eps)) #логарифм округленной оценки вероятности

        self.mask = np.zeros( (self.FEATURE_DIM), dtype=np.float32 ) # mask - z vector
        self.x    = np.zeros( (self.FEATURE_DIM), dtype=np.float32 ) # sample features
        self.y    = 0        
        
        min_value = kargs['MIN_VALUE'] if 'MIN_VALUE' in kargs else np.finfo(np.float32).min 
        max_value = kargs['MAX_VALUE'] if 'MAX_VALUE' in kargs else np.finfo(np.float32).max
        self.action_space = spaces.Discrete(self.ACTION_DIM)
        self.observation_space = spaces.Box(low=min_value, high=max_value, 
                                            shape=(2 * self.FEATURE_DIM,), dtype=np.float32)

        self.random_mode = random_mode
        self.idx = -1 #индекс текущего экземпляра, определяющего состояние среды
    
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

        return (s_next, r, done, {})   # state, reward, terminal flag, unavailable actions insted of info dict
    
    def reset(self):
        
        self.mask[:] = 0
        if self.random_mode:
            self.idx = np.random.randint(0, self.DATA_LEN)
        else: #идем по порядку
            self.idx = (self.idx + 1) % self.DATA_LEN
        self.x, self.y= self._generate_sample()
        s  = self._get_state(self.x, self.mask)
        return s
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        pass
    
    def _generate_sample(self):
        x = self.data_x[self.idx]        # sample features
        y = self.data_y[self.idx]        # class
        return (x, y)
    
    @staticmethod
    def _get_state(x, m):   #генерация наблюдаемого для агента состояния
        x_ = (x * m)        #скрытие неизвестных признаков
        s = np.concatenate( (x_, m), axis=0).astype(np.float32)
        return s
if __name__=="__main__":
    import numpy as np
    import gym
    import os
    # ============================================DATASET
    from gym_pcwcf.config.config_synthetic_simple import DATASET_SETTING, DATA_LOAD_FN, MODEL_PATH
    FEATURE_DIM = DATASET_SETTING['FEATURE_DIM']
    TERMINAL_ACTIONS = DATASET_SETTING['TERMINAL_ACTIONS']
    ACTION_DIM = DATASET_SETTING['ACTION_DIM']
    TRAIN_DATA_LEN = DATASET_SETTING['TRAIN_DATA_LEN']
    GYM_VERSION_NAME = 'gym_pcwcf:pcwcf-v0'
    # ============================================/DATASET
    env = gym.make(GYM_VERSION_NAME, data_load_fn=DATA_LOAD_FN,
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