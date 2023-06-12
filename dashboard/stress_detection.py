import pandas as pd
import numpy as np
import scipy.stats as stats
import os
from sklearn import preprocessing


def read_raw_data(id):
    pid = id[0] + '0' + id[1]

    file_path = "dashboard\\labelled_data\\"

    for f in os.listdir(file_path):
        if f.startswith(str(pid)) and f.endswith("EDA.csv"):
            eda_path = f
        if f.startswith(str(pid)) and f.endswith("BVP.csv"):
            bvp_path = f

    eda_data_raw = pd.read_csv(file_path + eda_path)
    x1 = np.linspace(0, 15, (eda_data_raw['eda'].shape[0]))
    x1 = pd.DataFrame(x1, columns=['time'])

    df_eda = pd.concat([eda_data_raw['eda'], x1],axis=1)

    #df_eda = eda_data_raw

    bvp_data_raw = pd.read_csv(file_path + bvp_path)
    x2 = np.linspace(0, 15, (bvp_data_raw['bvp'].shape[0]))
    x2 = pd.DataFrame(x2, columns=['time'])

    df_bvp = pd.concat([bvp_data_raw['bvp'], x2],axis=1)

    return df_eda, df_bvp

def read_processed_data(id):
    print(id)
    file_path = 'dashboard\\processed_data\\'
                       
    for f in os.listdir(file_path):
        if f.startswith(str(id)) and f.endswith("eda.csv"):
            eda_path = f
        elif f.startswith(str(id)) and f.endswith("temp.csv"):
            temp_path = f
        elif f.startswith(str(id)) and f.endswith("bvp.csv"):
            bvp_path = f
        elif f.startswith(str(id)) and f.endswith("acc_x.csv"):
            acc_x_path = f
        elif f.startswith(str(id)) and f.endswith("acc_y.csv"):
            acc_y_path = f
        elif f.startswith(str(id)) and f.endswith("acc_z.csv"):
            acc_z_path = f
        elif f.startswith(str(id)) and f.endswith("label.csv"):
            label_path = f

    eda_data_raw = pd.read_csv(file_path + eda_path)
    temp_data_raw = pd.read_csv(file_path + temp_path)
    bvp_data_raw = pd.read_csv(file_path + bvp_path)
    acc_x_data_raw = pd.read_csv(file_path + acc_x_path)
    acc_y_data_raw = pd.read_csv(file_path + acc_y_path)
    acc_z_data_raw = pd.read_csv(file_path + acc_z_path)
    label_data_raw = pd.read_csv(file_path + label_path)

    return {'eda':np.array(eda_data_raw, dtype=object),
            "temp": np.array(temp_data_raw, dtype=object),
            "bvp": np.array(bvp_data_raw, dtype=object),
            "acc_x": np.array(acc_x_data_raw, dtype=object),
            "acc_y": np.array(acc_y_data_raw, dtype=object),
            "acc_z": np.array(acc_z_data_raw, dtype=object),
            "label": np.array(label_data_raw)}


def prepare_data(data):
    
    X_test = []

    for feature in ['eda','bvp','temp','acc_x','acc_y','acc_z']:
        X_test_reshape = data[feature].reshape(data[feature].shape[0],data[feature].shape[1], 1).astype(np.float32)
        X_test.append(X_test_reshape)

    return X_test