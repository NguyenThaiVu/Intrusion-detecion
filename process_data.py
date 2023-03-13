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


def Process_Data_Type(df):
    data = df.copy()

    for column in data.columns:
        if data[column].dtype == np.int64:
            maxVal = data[column].max()
            if maxVal < 120:
                data[column] = data[column].astype(np.int8)
            elif maxVal < 32767:
                data[column] = data[column].astype(np.int16)
            else:
                data[column] = data[column].astype(np.int32)
                
        if data[column].dtype == np.float64:
            maxVal = data[column].max()
            minVal = data[data[column]>0][column]
            if maxVal < 120 and minVal>0.01 :
                data[column] = data[column].astype(np.float16)
            else:
                data[column] = data[column].astype(np.float32)

    return data


def Process_Label(data, is_binary_classifier=False, except_attack_type=None):

    attackType = data[' Label'].unique()    
    if is_binary_classifier == False:
        data[' Label'] = data[' Label'].astype('category')
        data[' Label'] = data[' Label'].astype("category").cat.codes
    else:
        if except_attack_type == None:
            data[' Label'] = data[' Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
            return data
        else:

            # ['BENIGN' 'DDoS' 'PortScan' 'Bot' 'Infiltration' 'Web Attack � Brute Force' 
            # 'Web Attack � XSS' 'Web Attack � Sql Injection']
            print(Counter(data[' Label']))

            # Extract just one type of attack
            except_attack = data[data[' Label'] == except_attack_type]
            except_attack = except_attack.reset_index(drop=False) 
            except_attack[' Label'] = except_attack[' Label'].apply(lambda x: '1' if x == except_attack_type else '0')
            except_attack[' Label'] = except_attack[' Label'].astype(np.float32)

            # Get other attack
            data = data[data[' Label'] != except_attack_type]
            data = data.reset_index(drop=False)
            data[' Label'] = data[' Label'].apply(lambda x: '0' if x == 'BENIGN' else '1')
            data[' Label'] = data[' Label'].astype(np.float32)

            return (data, except_attack)
    


def Handle_ImBalance(X_train, Y_train, list_feature_names):

    df = pd.DataFrame(X_train, columns=list_feature_names)
    df[' Label'] = np.array(Y_train)

    minor = pd.DataFrame(df[(df[' Label']!=0) & (df[' Label']!=2)])
    major = pd.DataFrame(df[(df[' Label']==0) | (df[' Label']==2)])

    # Upsampling minor classes
    y_rus_ =  minor[' Label']
    X_rus_ =  minor.drop([' Label'],axis=1)
    strategy = {1: 10000, 3: 10000, 4: 10000, 5: 10000, 6: 10000, 7: 10000}
    sm = SMOTE(sampling_strategy=strategy)
    X_sm, y_sm = sm.fit_resample(X_rus_, y_rus_)
    X_min,y_min = X_sm, y_sm 

    # Undersampling major classes
    y_rus_ =  major[' Label']
    X_rus_ =  major.drop([' Label'],axis=1)
    strategy = {0:10000, 2:10000}
    tom = RandomUnderSampler(sampling_strategy=strategy)
    X_tom, y_tom = tom.fit_resample(X_rus_, y_rus_)

    # Get final result
    X_maj,y_maj = X_tom, y_tom
    X_train, Y_train = pd.concat([X_maj,X_min]), pd.concat([y_maj,y_min])

    return (X_train, Y_train)



def Prepare_Data_Multiple_Classes(data):

    y = data[' Label'].copy()
    X = data.drop([' Label'],axis=1)
    list_feature_names = X.columns

    #  Train test split
    X_train,X_test,Y_train,Y_test = train_test_split(X, y,train_size=0.70, random_state=2)

    # 3. Handle imbalance dataset - oversampling
    (X_train, Y_train) = Handle_ImBalance(X_train, Y_train, list_feature_names)
 
    # Scale numerical features to have zero mean and unit variance  
    scaler = StandardScaler()
    cols = X_train.select_dtypes(include=['float32','float16','int32','int16','int8']).columns
    X_train = scaler.fit_transform(X_train.select_dtypes(include=['float32','float16','int32','int16','int8']))
    X_test = scaler.transform(X_test.select_dtypes(include=['float32','float16','int32','int16','int8']))

    # Convert label to one-one encoding
    lb = LabelBinarizer()
    Y_train = lb.fit_transform(Y_train)
    Y_test = lb.transform(Y_test)

    return (X_train, X_test, Y_train, Y_test)


def Prepare_Data_Binary(data):
    if "index" in data.columns:
        data = data.drop(['index'],axis=1)

    y = data[' Label'].copy()
    X = data.drop([' Label'],axis=1)
    list_feature_names = X.columns

    #  Train test split
    X_train,X_test,Y_train,Y_test = train_test_split(X, y,train_size=0.70, random_state=2)

    # Scale numerical features to have zero mean and unit variance  
    scaler = StandardScaler()
    cols = X_train.select_dtypes(include=['float32','float16','int32','int16','int8']).columns
    X_train = scaler.fit_transform(X_train.select_dtypes(include=['float32','float16','int32','int16','int8']))
    X_test = scaler.transform(X_test.select_dtypes(include=['float32','float16','int32','int16','int8']))

    return (X_train, X_test, Y_train, Y_test)

