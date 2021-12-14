import numpy as np
import gym
import os
#============================================DATASET
from gym_pcwcf.config.config_cicids2017_webattacks import DATASET_SETTING, DATA_VAL_LOAD_FN, DATA_TEST_LOAD_FN
#============================================/DATASET
GYM_VERSION_NAME = 'gym_pcwcf:cwcf-v0'
IS_BINARY_CLASSIFICATION = True
LAMBDA = 1e-4
env_kwargs = dict( dataset_name='cicids2017_webattacks',
                   lambda_coefficient=LAMBDA,
                   is_binary_classification=IS_BINARY_CLASSIFICATION)
print(f'current directory is {os.getcwd()}')
FEATURE_DIM     = DATASET_SETTING['FEATURE_DIM']
CLASS_DIM = DATASET_SETTING['CLASS_DIM']
TERMINAL_ACTIONS = 2 if IS_BINARY_CLASSIFICATION==True else CLASS_DIM
ACTION_DIM      = FEATURE_DIM + TERMINAL_ACTIONS
TRAIN_DATA_LEN  = DATASET_SETTING['TRAIN_DATA_LEN']

def test_common_check():
     #common checks - type and size
    env = gym.make(GYM_VERSION_NAME, **env_kwargs)
    np.testing.assert_equal(2 * FEATURE_DIM , env.observation_space.shape[0])
    np.testing.assert_equal(env.action_space.n, ACTION_DIM)
    np.testing.assert_equal(env.DATA_LEN, TRAIN_DATA_LEN)
       
    s = env.reset()
    na = np.concatenate( (np.zeros((TERMINAL_ACTIONS)),
                          s[int(s.shape[0]/2):] ) )
    
    assert type(s) is np.ndarray
    assert type(na) is np.ndarray
    
    assert np.any(np.equal(s, np.zeros(2 * FEATURE_DIM)))
    assert np.any(np.equal(na, np.zeros(ACTION_DIM, dtype = np.bool)))
    
def test_randomness():
    env_1 = gym.make(GYM_VERSION_NAME, **env_kwargs)
    env_1.reset()
    s1 = env_1.x
    env_2 = gym.make(GYM_VERSION_NAME, **env_kwargs)
    env_2.reset()
    s2 = env_2.x
    env_2.reset()
    s3 = env_2.x

    assert not np.all(np.equal(s1, s2)), "initial states should be different for each environment"
    assert not np.all(np.equal(s2, s3)), "initial states should be different  after reset environment"

def test_determinism_for_random_mode_test_dataset():
    #also test TEST mode
    env = gym.make(GYM_VERSION_NAME,random_mode=False, mode='TEST', **env_kwargs)
    np.testing.assert_equal(env.DATA_LEN, DATASET_SETTING['TEST_DATA_LEN'])
    env.reset()
    x0 = env.x
    y0 = env.y
    env.reset()
    x1 = env.x
    y1 = env.y
    data = DATA_TEST_LOAD_FN()[:2]
    data_x, data_y = data[:, 0:-1].astype('float32'), data[:, -1].astype('int32')
    assert np.all(np.equal(x0, data_x[0]))
    assert np.all(np.equal(x1, data_x[1]))
    assert np.equal(y0, data_y[0])
    assert np.equal(y1, data_y[1])

def test_determinism_for_random_mode_val_dataset():
    #also test TEST mode
    env = gym.make(GYM_VERSION_NAME,random_mode=False, mode='VAL', **env_kwargs)
    np.testing.assert_equal(env.DATA_LEN, DATASET_SETTING['VAL_DATA_LEN'])
    env.reset()
    x0 = env.x
    y0 = env.y
    env.reset()
    x1 = env.x
    y1 = env.y
    data = DATA_VAL_LOAD_FN()[:2]
    data_x, data_y = data[:, 0:-1].astype('float32'), data[:, -1].astype('int32')
    assert np.all(np.equal(x0, data_x[0]))
    assert np.all(np.equal(x1, data_x[1]))
    assert np.equal(y0, data_y[0])
    assert np.equal(y1, data_y[1])
    
def test_one_step():
    #check observation after one step
    env = gym.make(GYM_VERSION_NAME, **env_kwargs)
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
    env = gym.make(GYM_VERSION_NAME, **env_kwargs)
    s = env.reset()
    x_previous = env.x
    
    end_action = np.random.randint(TERMINAL_ACTIONS)
    s_next, r, done, _ = env.step(end_action)
    
    assert done == True
    assert np.all(env.x == x_previous)
    assert np.all(s == s_next)

def test_reward():
    env = gym.make(GYM_VERSION_NAME, **env_kwargs)
    env.reset()
    x, y = env.x, env.y
    target_rewards = np.ones(ACTION_DIM - TERMINAL_ACTIONS + 1) * -LAMBDA
    true_rewards = np.array( [env.step(action)[1] for action in range(ACTION_DIM-1, TERMINAL_ACTIONS-1, -1)] )
    terminal_action = y
    target_rewards[-1] = 0
    terminal_reward = env.step(terminal_action)[1]
    true_rewards = np.append(true_rewards, terminal_reward)
    np.testing.assert_almost_equal(true_rewards, target_rewards, decimal=14)

    env.reset()
    x, y = env.x, env.y
    target_rewards = np.ones(ACTION_DIM - TERMINAL_ACTIONS + 1) * -LAMBDA
    true_rewards = np.array( [env.step(action)[1] for action in range(ACTION_DIM-1, TERMINAL_ACTIONS-1, -1)] )
    terminal_action = (y+1)%env.TERMINAL_ACTIONS
    target_rewards[-1] = -1
    terminal_reward = env.step(terminal_action)[1]
    true_rewards = np.append(true_rewards, terminal_reward)
    np.testing.assert_almost_equal(true_rewards, target_rewards, decimal=14)

    env.reset()
    x, y = env.x, env.y
    target_rewards = np.ones(TERMINAL_ACTIONS) * -1
    target_rewards[y] = 0 #rigth classification
    true_rewards = np.array([env.step(terminal_action)[1] for terminal_action in np.arange(TERMINAL_ACTIONS)])
    np.testing.assert_equal(true_rewards, target_rewards)

def test_by_stablebaseline():
    from stable_baselines3.common.env_checker import check_env
    env = gym.make(GYM_VERSION_NAME, **env_kwargs)
    try:
        check_env(env)
    except Exception as e:
        # Just print(e) is cleaner and more likely what you want,
        # but if you insist on printing message specifically whenever possible...
        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)


def test_actions_mask():
    env = gym.make(GYM_VERSION_NAME, **env_kwargs)
    env.reset()
    env.step(TERMINAL_ACTIONS )
    env.step(TERMINAL_ACTIONS + 1)
    true_actions_mask = env.action_masks()
    target_actions_mask = np.ones(ACTION_DIM)
    target_actions_mask[TERMINAL_ACTIONS] = 0
    target_actions_mask[TERMINAL_ACTIONS+1] = 0
    np.testing.assert_equal(true_actions_mask, target_actions_mask)

def test_known_feature():
    env = gym.make(GYM_VERSION_NAME,**env_kwargs)
    env.reset()
    action = np.random.randint(CLASS_DIM, ACTION_DIM)
    try:
        env.step(action)
        env.step(action)
        assert False
    except ValueError as e:
        assert True
if __name__=='__main__':
    test_by_stablebaseline()