import numpy as np

DATASET = 'synthetic-simple'
DATA_PATH = '/home/student/projects/data/cicids2017simple/data-'
EXTENSION = '.npy'
DATA_TRN_PATH = DATA_PATH + 'train' + EXTENSION
DATA_VAL_PATH = DATA_PATH + 'val' + EXTENSION
DATA_TEST_PATH = DATA_PATH + 'test' + EXTENSION
DATA_TRAIN_LOAD_FN = lambda : np.load(DATA_TRN_PATH)
DATA_VAL_LOAD_FN = lambda : np.load(DATA_VAL_PATH)
DATA_TEST_LOAD_FN = lambda : np.load(DATA_TEST_PATH, allow_pickle=True)
MODEL_PATH = '/home/student/projects/data/synthetic_pcwcf/models/synthetic-simple-model'

FEATURES =  ['Total Length of Bwd Packets',
             'Average Packet Size',
             'Subflow Fwd Bytes',
             'Total Length of Fwd Packets',
             'Subflow Bwd Bytes',
             'Destination Port',
             'Avg Fwd Segment Size',
             'Fwd Packet Length Mean',
             'Bwd Packets/s',
             'Max Packet Length']

ATTACK_MAPPING =    {0: 'BENIGN',
                     1: 'DDoS',
                     2: 'DoS GoldenEye',
                     3: 'DoS Hulk',
                     4: 'DoS Slowhttptest',
                     5: 'DoS slowloris',
                     6: 'Heartbleed'}

DATASET_SETTING = {
    'CLASS_DIM'         :   len(ATTACK_MAPPING),
    'FEATURE_DIM'       :   len(FEATURES),
    'TRAIN_DATA_LEN'    : 733948
}
