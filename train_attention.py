import sys
sys.dont_write_bytecode = True
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import pandas as pd
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[1], 'GPU')

from process_data import *
from model import *
from evaluation import *

# --------------- Hyper parameter ---------------------
N_features = 78
N_labels = 1

BATCH_SIZE = 128
NUM_EPOCHS = 3

is_binary_classifier = True
unseen_attack_type = 'PortScan'
# -----------------------------------------------------

def main():

    data = pd.read_csv(r'data/intrusion_detection_dataset.csv')
    data = Remove_INF_and_NaN(data)

    if unseen_attack_type != None:
        (data, unseen_attack) = Process_Label(data, is_binary_classifier=is_binary_classifier, \
                                            unseen_attack_type=unseen_attack_type)
    else:
        data = Process_Label(data, is_binary_classifier=True)

    (X_train, X_test, Y_train, Y_test, scaler) = Prepare_Data_Binary(data, is_handle_imbalance=True)

    # Save for second round training
    (X_train, X_train_2, Y_train, Y_train_2) = train_test_split(X_train, Y_train, test_size=0.1)
    

    print("\n-------------------------- Training on original dataset --------------------------")
    print("Number of samples per class: {}".format(Counter(Y_train)))
    print("X_train: {}".format(X_train.shape))
    print("Y_train: {}".format(Y_train.shape))
    print('----------------------------------------\n')

    model = Define_Attention_Model(N_features, N_labels)
    model.fit(X_train, Y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)


    print("\n-------------------------- Evaluate on testing dataset --------------------------")
    Evaluate_Model(model, X_test, Y_test, is_binary_classifier=True)

    
    print("\n-------------------------- Evaluate on unseen dataset --------------------------")
    if "index" in data.columns:
        data = data.drop(['index'],axis=1)
    Y_unseen_attack = unseen_attack[' Label'].copy()
    X_unseen_attack = unseen_attack.drop([' Label'],axis=1)
    X_unseen_attack = scaler.transform(X_unseen_attack)

    Evaluate_Model(model, X_unseen_attack, Y_unseen_attack, is_binary_classifier=True)


    print("\n-------------------------- Transfer learning on SECOND round -------------------------- ")
    X_train_2 = np.concatenate([X_unseen_attack, X_train_2], axis=0)
    Y_train_2 = np.concatenate([Y_unseen_attack, Y_train_2], axis=0)
    
    print("X_train 2: {}".format(X_train_2.shape))
    print("Y_train 2: {}".format(Y_train_2.shape))
    print("Number of samples per class: {}".format(Counter(Y_unseen_attack)))

    model.fit(X_train_2, Y_train_2, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)


    print("\n-------------------------- Evaluate AGAIN unseen dataset --------------------------")
    Evaluate_Model(model, X_unseen_attack, Y_unseen_attack, is_binary_classifier=True)

    print("\n-------------------------- Evaluate AGAIN on testing dataset --------------------------")
    Evaluate_Model(model, X_test, Y_test, is_binary_classifier=True)


main()