import sys
sys.dont_write_bytecode = True
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import pandas as pd
import tensorflow as tf
# gpus = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(gpus[1], 'GPU')

from process_data import *
from model import *
from evaluation import *

# --------------- Hyper parameter ---------------------
N_features = 12
N_labels = 1

BATCH_SIZE = 128
NUM_EPOCHS = 5

is_binary_classifier = True
except_attack_type = "Bot"

# -----------------------------------------------------

def main():

    df = pd.read_csv(r'data/intrusion_detection_dataset.csv')
    df = Remove_INF_and_NaN(df)

    data = Process_Data_Type(df)

    if except_attack_type != None:
        (data, except_attack) = Process_Label(data, is_binary_classifier=True, except_attack_type=except_attack_type)
    else:
        data = Process_Label(data, is_binary_classifier=True)


    (X_train, X_test, Y_train, Y_test) = Prepare_Data_Binary(data)
    print("----- [INFO] Finish prepare data -----")
    print("Number of samples per class: {}".format(Counter(Y_train)))
    print("X_train: {}".format(X_train.shape))
    print("Y_train: {}".format(Y_train.shape))
    print("Except attack shape: {}".format(except_attack.shape))
    print('----------------------------------------')


    model = Define_Attention_Model(N_features, N_labels)

    model.fit(X_train, Y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

    save_model(model, 'my_model.h5')

    Evaluate_Model(model, X_test, Y_test, is_binary_classifier=True)

    Evaluate_Model_DDos_Attack(model, except_attack)


main()