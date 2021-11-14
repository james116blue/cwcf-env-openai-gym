"""
global config file
"""
import numpy as np

DATASET = 'synthetic-simple'
DATA_PATH = '/home/student/projects/data/synthetic_pcwcf/synthetic-simple-'
EXTENSION = '.npy'
DATA_TRN_PATH = DATA_PATH + 'train' + EXTENSION
DATA_VAL_PATH = DATA_PATH + 'val' + EXTENSION
DATA_TEST_PATH = DATA_PATH + 'test' + EXTENSION
DATA_LOAD_FN = lambda : np.load(DATA_TRN_PATH)
DATA_VAL_LOAD_FN = lambda : np.load(DATA_VAL_PATH)
DATA_TEST_LOAD_FN = lambda : np.load(DATA_TEST_PATH)
MODEL_PATH = '/home/student/projects/data/synthetic_pcwcf/models/synthetic-simple-model'
DATASET_SETTING = {
    'CLASS_DIM'  : 2,
    'FEATURE_DIM'       : 5,
    'TRAIN_DATA_LEN'         : 1024
}
