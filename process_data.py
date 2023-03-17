import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def Remove_INF_and_NaN(df):

    numerical_cols = df.select_dtypes(include=['number']).columns
    for column_name in numerical_cols:
        df = df[~np.isinf(df[column_name])]
    
    df = df.dropna(axis=1)

    df = df.reset_index(drop=False) # To make sure the index is right 
    return df


def Process_Label(data, is_binary_classifier=False, unseen_attack_type=None):

    if "index" in data.columns: data = data.drop(['index'],axis=1)

    if is_binary_classifier == False:
        print(Counter(data[' Label']))
        data[' Label'] = data[' Label'].astype('category')
        data[' Label'] = data[' Label'].astype("category").cat.codes
    else:
        if unseen_attack_type == None:
            data[' Label'] = data[' Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
            data[' Label'] = data[' Label'].astype(np.float32)
            return data
        else:
            # ['BENIGN' 'DDoS' 'PortScan' 'Bot'....]
            # Extract just one type of attack
            unseen_attack = data[data[' Label'] == unseen_attack_type]
            unseen_attack = unseen_attack.reset_index(drop=False)
            unseen_attack[' Label'] = unseen_attack[' Label'].apply(lambda x: 1 if x == unseen_attack_type else 0)
            unseen_attack[' Label'] = unseen_attack[' Label'].astype(np.float32)
            if "index" in unseen_attack.columns: unseen_attack = unseen_attack.drop(['index'],axis=1)

            print("Dataframe shape unseen_attack: {}".format(unseen_attack.shape))


            # Get other remain attack type
            data = data[data[' Label'] != unseen_attack_type]
            data = data.reset_index(drop=False)
            data[' Label'] = data[' Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
            data[' Label'] = data[' Label'].astype(np.float32)
            if "index" in data.columns: data = data.drop(['index'],axis=1)

            print("Dataframe shape remain data: {}".format(data.shape))

            return (data, unseen_attack)
    


def Handle_ImBalance(X, Y):

    print("Before sampling: ", Counter(Y))

    # Undersample the majority class
    rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    X, Y = rus.fit_resample(X, Y)

    # Oversample the minority class
    ros = RandomOverSampler(sampling_strategy='minority', random_state=42)
    X, Y = ros.fit_resample(X, Y)

    print("After sampling: ", Counter(Y))
    return (X, Y)



def Prepare_Data_Binary(data, is_handle_imbalance=False, is_scale=True):
    if "index" in data.columns:
        data = data.drop(['index'],axis=1)

    y = data[' Label'].copy()
    X = data.drop([' Label'],axis=1)
    list_feature_names = X.columns

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

    if is_handle_imbalance == True:
        (X_train, Y_train) = Handle_ImBalance(X_train, Y_train)

    # Scale numerical features to have zero mean and unit variance  
    scaler = None
    if is_scale == True:
        scaler = StandardScaler()
        scaler = scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    return (X_train, X_test, Y_train, Y_test, scaler)
