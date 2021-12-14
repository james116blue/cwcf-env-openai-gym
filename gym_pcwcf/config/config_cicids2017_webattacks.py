import numpy as np

DATASET = 'cicids2017_webattacks'
DATA_PATH = '/home/student/projects/data/cicids2017_webattacks/data-'
EXTENSION = '.npy'
DATA_TRN_PATH = DATA_PATH + 'train' + EXTENSION
DATA_VAL_PATH = DATA_PATH + 'val' + EXTENSION
DATA_TEST_PATH = DATA_PATH + 'test' + EXTENSION
DATA_TRAIN_LOAD_FN = lambda : np.load(DATA_TRN_PATH)
DATA_VAL_LOAD_FN = lambda : np.load(DATA_VAL_PATH)
DATA_TEST_LOAD_FN = lambda : np.load(DATA_TEST_PATH, allow_pickle=True)
MODEL_PATH = '/home/student/projects/data/synthetic_pcwcf/models/synthetic-simple-model'

FEATURES =  {0: 'ACK Flag Count',
             1: 'Active Max',
             2: 'Active Mean',
             3: 'Active Min',
             4: 'Active Std',
             5: 'Average Packet Size',
             6: 'Avg Bwd Segment Size',
             7: 'Avg Fwd Segment Size',
             8: 'Bwd Header Length',
             9: 'Bwd IAT Max',
             10: 'Bwd IAT Mean',
             11: 'Bwd IAT Min',
             12: 'Bwd IAT Std',
             13: 'Bwd IAT Total',
             14: 'Bwd Packet Length Max',
             15: 'Bwd Packet Length Mean',
             16: 'Bwd Packet Length Min',
             17: 'Bwd Packet Length Std',
             18: 'Bwd Packets/s',
             19: 'Down/Up Ratio',
             20: 'ECE Flag Count',
             21: 'FIN Flag Count',
             22: 'Flow Bytes/s',
             23: 'Flow Duration',
             24: 'Flow IAT Max',
             25: 'Flow IAT Mean',
             26: 'Flow IAT Min',
             27: 'Flow IAT Std',
             28: 'Flow Packets/s',
             29: 'Fwd Header Length',
             30: 'Fwd IAT Max',
             31: 'Fwd IAT Mean',
             32: 'Fwd IAT Min',
             33: 'Fwd IAT Std',
             34: 'Fwd IAT Total',
             35: 'Fwd PSH Flags',
             36: 'Fwd Packet Length Max',
             37: 'Fwd Packet Length Mean',
             38: 'Fwd Packet Length Min',
             39: 'Fwd Packet Length Std',
             40: 'Fwd Packets/s',
             41: 'Idle Max',
             42: 'Idle Mean',
             43: 'Idle Min',
             44: 'Idle Std',
             45: 'Max Packet Length',
             46: 'Min Packet Length',
             47: 'PSH Flag Count',
             48: 'Packet Length Mean',
             49: 'Packet Length Std',
             50: 'Packet Length Variance',
             51: 'RST Flag Count',
             52: 'SYN Flag Count',
             53: 'Subflow Bwd Bytes',
             54: 'Subflow Bwd Packets',
             55: 'Subflow Fwd Bytes',
             56: 'Subflow Fwd Packets',
             57: 'Total Backward Packets',
             58: 'Total Fwd Packets',
             59: 'Total Length of Bwd Packets',
             60: 'Total Length of Fwd Packets',
             61: 'URG Flag Count',
             62: 'act_data_pkt_fwd',
             63: 'min_seg_size_forward'}

ATTACKS =      {0: 'BENIGN',
                1: 'Web Attack � Brute Force',
                2: 'Web Attack � Sql Injection',
                3: 'Web Attack � XSS'}

DATASET_SETTING = {
    'CLASS_DIM'         :   len(ATTACKS),
    'FEATURE_DIM'       :   len(FEATURES),
    'TRAIN_DATA_LEN'    : 4096,
    'VAL_DATA_LEN'      : 990,
    'TEST_DATA_LEN'     : 2181,
}
