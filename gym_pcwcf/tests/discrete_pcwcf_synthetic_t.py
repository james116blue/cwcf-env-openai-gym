import numpy as np
import gym
import os
#============================================DATASET
from gym_pcwcf.config.config_synthetic_simple import DATASET_SETTING, DATA_LOAD_FN, MODEL_PATH
#============================================/DATASET
GYM_VERSION_NAME = 'gym_pcwcf:pcwcf-v0'

print(f'current directory is {os.getcwd()}')
FEATURE_DIM     = DATASET_SETTING['FEATURE_DIM']
TERMINAL_ACTIONS= 11
ACTION_DIM      = FEATURE_DIM + TERMINAL_ACTIONS
TRAIN_DATA_LEN  = DATASET_SETTING['TRAIN_DATA_LEN']

def test_common_check():
     #common checks - type and size
    env = gym.make(GYM_VERSION_NAME, data_load_fn=DATA_LOAD_FN,
                   lambda_coefficient=1e-3,
                   costs=None,
                   discretization_class_number=11,
                   random_mode=True,
                   eps=1e-15)
    assert np.equal(2 * FEATURE_DIM , env.observation_space.shape[0])
    assert np.equal(ACTION_DIM, env.action_space.n)
    assert np.equal(env.DATA_LEN, TRAIN_DATA_LEN)
       
    s = env.reset()
    na = np.concatenate( (np.zeros((TERMINAL_ACTIONS)),
                          s[int(s.shape[0]/2):] ) )
    
    assert type(s) is np.ndarray
    assert type(na) is np.ndarray
    
    assert np.any(np.equal(s, np.zeros(2 * FEATURE_DIM)))
    assert np.any(np.equal(na, np.zeros(ACTION_DIM, dtype = np.bool)))
    
def test_randomness():
    env_1 = gym.make(GYM_VERSION_NAME, data_load_fn=DATA_LOAD_FN,
                   lambda_coefficient=1e-3,
                   costs=None,
                   discretization_class_number=11,
                   random_mode=True,
                   eps=1e-15)
    env_1.reset()
    s1 = env_1.x
    env_2 = gym.make(GYM_VERSION_NAME, data_load_fn=DATA_LOAD_FN,
                   lambda_coefficient=1e-3,
                   costs=None,
                   discretization_class_number=11,
                   random_mode=True,
                   eps=1e-15)
    env_2.reset()
    s2 = env_2.x
    env_2.reset()
    s3 = env_2.x

    assert not np.all(np.equal(s1, s2)), "initial states should be different for each environment"
    assert not np.all(np.equal(s2, s3)), "initial states should be different  after reset environment"

def test_determinism_for_random_mode():
    env = gym.make(GYM_VERSION_NAME, data_load_fn=DATA_LOAD_FN,
                   lambda_coefficient=1e-3,
                   costs=None,
                   discretization_class_number=11,
                   random_mode=False,
                   eps=1e-15)
    env.reset()
    x0 = env.x
    y0 = env.y
    env.reset()
    x1 = env.x
    y1 = env.y
    data = DATA_LOAD_FN()[:2]
    data_x, data_y = data[:, 0:-1].astype('float32'), data[:, -1].astype('int32')
    assert np.all(np.equal(x0, data_x[0]))
    assert np.all(np.equal(x1, data_x[1]))
    assert np.equal(y0, data_y[0])
    assert np.equal(y1, data_y[1])
    
def test_one_step():
    #check observation after one step
    env = gym.make(GYM_VERSION_NAME, data_load_fn=DATA_LOAD_FN,
                   lambda_coefficient=1e-3,
                   costs=None,
                   discretization_class_number=11,
                   random_mode=True,
                   eps=1e-15)
    s = env.reset()
    na = np.concatenate( (np.zeros((TERMINAL_ACTIONS)),
                          s[int(s.shape[0]/2):] ) )
    
    action = np.random.randint(TERMINAL_ACTIONS, ACTION_DIM)
    (s_next, r, done, _) = env.step(action)
    state = s_next[:FEATURE_DIM]
    mask = s_next[FEATURE_DIM:]
    assert done == False
    #check state
    assert state[action-TERMINAL_ACTIONS] == env.x[action-TERMINAL_ACTIONS]   
    assert np.all(state[np.arange(len(state))!=action-TERMINAL_ACTIONS] == 0)
    #check mask
    assert mask[action-TERMINAL_ACTIONS] == 1  
    assert np.all(mask[np.arange(len(mask))!=action-TERMINAL_ACTIONS] == 0)
    #unavailabel actions
    assert np.all(na[np.arange(len(na))!=action] == False)     #all except one is False
    
def test_last_step():
    #check end step of episode 
    env = gym.make(GYM_VERSION_NAME, data_load_fn=DATA_LOAD_FN,
                   lambda_coefficient=1e-3,
                   costs=None,
                   discretization_class_number=11,
                   random_mode=True,
                   eps=1e-15)
    s = env.reset()
    x_previous = env.x
    
    end_action = np.random.randint(TERMINAL_ACTIONS)
    s_next, r, done, _ = env.step(end_action)
    
    assert done == True
    assert np.all(env.x == x_previous)
    assert np.all(s == s_next)

def test_reward():
    env = gym.make(GYM_VERSION_NAME, data_load_fn=DATA_LOAD_FN,
                   lambda_coefficient=1e-3,
                   costs=None,
                   discretization_class_number=11,
                   random_mode=True,
                   eps=1e-15)
    env.reset()
    x, y = env.x, env.y
    target_rewards = np.ones(ACTION_DIM - TERMINAL_ACTIONS + 1) * -0.001
    true_rewards = np.array( [env.step(action)[1] for action in range(ACTION_DIM-1, TERMINAL_ACTIONS-1, -1)] )
    terminal_action = 8
    target_rewards[-1] = np.log(0.8) if y==1 else np.log(0.2)
    terminal_reward = env.step(terminal_action)[1]
    true_rewards = np.append(true_rewards, terminal_reward)
    np.testing.assert_almost_equal(true_rewards, target_rewards, decimal=14)

    env.reset()
    x, y = env.x, env.y
    target_rewards = np.ones(ACTION_DIM - TERMINAL_ACTIONS + 1) * -0.001
    true_rewards = np.array( [env.step(action)[1] for action in range(ACTION_DIM-1, TERMINAL_ACTIONS-1, -1)] )
    terminal_action = 0
    target_rewards[-1] = np.log(1e-15) if y==1 else np.log(1 - 1e-15)
    terminal_reward = env.step(terminal_action)[1]
    true_rewards = np.append(true_rewards, terminal_reward)
    np.testing.assert_almost_equal(true_rewards, target_rewards, decimal=14)

def test_by_stablebaseline():
    from stable_baselines3.common.env_checker import check_env
    env = gym.make(GYM_VERSION_NAME, data_load_fn=DATA_LOAD_FN,
                   lambda_coefficient=1e-3,
                   costs=None,
                   discretization_class_number=11,
                   random_mode=True,
                   eps=1e-15)
    check_env(env)

def test_actions_mask():
    env = gym.make(GYM_VERSION_NAME, data_load_fn=DATA_LOAD_FN,
                   lambda_coefficient=1e-3,
                   costs=None,
                   discretization_class_number=11,
                   random_mode=True,
                   eps=1e-15)
    env.reset()
    env.step(TERMINAL_ACTIONS )
    env.step(TERMINAL_ACTIONS + 1)
    true_actions_mask = env.action_masks()
    target_actions_mask = np.ones(ACTION_DIM)
    target_actions_mask[TERMINAL_ACTIONS] = 0
    target_actions_mask[TERMINAL_ACTIONS+1] = 0
    np.testing.assert_equal(true_actions_mask, target_actions_mask)

env = gym.make(GYM_VERSION_NAME, data_load_fn=DATA_LOAD_FN,
               lambda_coefficient=1e-3,
               costs=None,
               discretization_class_number=11,
               random_mode=True,
               eps=1e-15)