import pandas as pd
import numpy as np
import scipy.stats as stats
import os
from sklearn import preprocessing

def read_raw_data(pid):

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

    bvp_data_raw = pd.read_csv(file_path + bvp_path)
    x2 = np.linspace(0, 15, (bvp_data_raw['bvp'].shape[0]))
    x2 = pd.DataFrame(x2, columns=['time'])

    df_bvp = pd.concat([bvp_data_raw['bvp'], x2],axis=1)

    return df_eda, df_bvp

def s(samples):
    
    std = np.std(samples)
    if std == 0:
        return samples - np.mean(samples)
    else:
        return (samples - np.mean(samples)) / std

def preprocess(df):
    scaler = preprocessing.StandardScaler() #MinMaxScaler
    df_drop = df.drop(['label'], axis=1)
    names = df_drop.columns
    fit = scaler.fit_transform(df_drop)
    scaled_df = pd.DataFrame(fit, columns=names)
    df_new = pd.concat([df['label'],scaled_df], axis = 1)
    return df_new

def read_processed_data(pid):

    eda_freq = 4
    temp_freq = 4
    bvp_freq = 64
    acc_freq = 32

    print(pid)
    file_path = 'dashboard\\labelled_data\\'
                       
    for f in os.listdir(file_path):
        if f.startswith(str(pid)) and f.endswith("EDA.csv"):
            eda_path = f
        elif f.startswith(str(pid)) and f.endswith("TEMP.csv"):
            temp_path = f
        elif f.startswith(str(pid)) and f.endswith("BVP.csv"):
            bvp_path = f
        elif f.startswith(str(pid)) and f.endswith("ACC.csv"):
            acc_path = f

    eda_data_raw = pd.read_csv(file_path + eda_path)
    temp_data_raw = pd.read_csv(file_path + temp_path)
    bvp_data_raw = pd.read_csv(file_path + bvp_path)
    acc_data_raw = pd.read_csv(file_path + acc_path)

    eda_data = preprocess(eda_data_raw)
    temp_data = preprocess(temp_data_raw)
    bvp_data = preprocess(bvp_data_raw)
    acc_data = preprocess(acc_data_raw)

    eda = []; temp=[]; label=[]; bvp=[]; hr=[]; eye=[]; win_size = 30; acc_x=[]; acc_y=[]; acc_z=[]
    
    for i in range(win_size, int(len(bvp_data_raw['bvp']) / bvp_freq)):

        eda.append(s(eda_data['eda'][eda_freq * (i - win_size): eda_freq * i]))

        temp.append(s(temp_data['temp'][temp_freq * (i - win_size): temp_freq * i]))

        bvp.append(s(bvp_data['bvp'][bvp_freq * (i - win_size): bvp_freq * i]))

        acc_x.append(s(acc_data['x'][acc_freq * (i - win_size): acc_freq * i]))

        acc_y.append(s(acc_data['y'][acc_freq * (i - win_size): acc_freq * i]))

        acc_z.append(s(acc_data['z'][acc_freq * (i - win_size): acc_freq * i]))

        l = eda_data["label"][eda_freq * (i - win_size): eda_freq * i]
        lp = np.bincount(np.array(l)).argmax()
        label.append(lp)

    return {"acc_x": np.array(acc_x, dtype=object),
            "acc_y": np.array(acc_y, dtype=object), 
            "acc_z": np.array(acc_z, dtype=object), 
            "eda": np.array(eda, dtype=object), 
            "temp": np.array(temp, dtype=object), 
            "bvp": np.array(bvp, dtype=object),
            "label": np.array(label)}


def prepare_data(data):
    
    X_test = []

    for feature in ['eda','bvp','temp','acc_x','acc_y','acc_z']:
        X_test_reshape = data[feature].reshape(data[feature].shape[0],data[feature].shape[1], 1).astype(np.float32)
        X_test.append(X_test_reshape)

    return X_test

def read_test_data(pid):
    
    X_test = {}

    test_files = "dashboard\\test_data\\[\'"
    all_features = ['eda','bvp','temp','acc_x','acc_y','acc_z']
    X_test_reshape = []

    for feature in all_features:
        X_test[feature] = pd.read_csv(test_files+str(pid)+"\']"+"_"+str(feature)+".csv", delimiter=',', header=None, index_col=False)
        X_test[feature] = np.array(X_test[feature])

        X_test[feature] = X_test[feature].reshape((X_test[feature].shape[0], X_test[feature].shape[1], 1)).astype(np.float32)

        #X_test_reshape = data[feature].reshape(data[feature].shape[0],data[feature].shape[1], 1).astype(np.float32)
        #X_test.append(X_test_reshape)

        X_test_reshape.append(X_test[feature])

    return X_test_reshape