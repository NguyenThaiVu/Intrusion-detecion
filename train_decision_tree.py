import sys
sys.dont_write_bytecode = True

import pandas as pd

from process_data import *
from model import *
from evaluation import *

# --------------- Hyper parameter ---------------------
N_features = 12
N_labels = 8
# -----------------------------------------------------

def main():

    df = pd.read_csv(r'data/intrusion_detection_dataset.csv')
    df = Remove_INF_and_NaN(df)

    data = Process_Data_Type(df, is_binary_classifier=True)

    (X_train, X_test, Y_train, Y_test) = Prepare_Data_Binary(data)
    print("[INFO] Finish prepare data")
    print("Number of samples per class: {}".format(Counter(Y_train)))
    print("X_train: {}".format(X_train.shape))
    print("Y_train: {}".format(Y_train.shape))
    
    model = Define_Random_Forest_Model()
    model.fit(X_train, Y_train)

    # Evaluation
    Y_pred =  model.predict(X_test)

    accuracy = metrics.accuracy_score(Y_test, Y_pred)
    confusion_matrix = metrics.confusion_matrix(Y_test, Y_pred)
    classification = metrics.classification_report(Y_test, Y_pred)
    print()
    print('============================== Model Evaluation ==============================')
    print()
    print ("Model Accuracy:" "\n", accuracy)
    print()
    print("Confusion matrix:" "\n", confusion_matrix)
    print()
    print("Classification report:" "\n", classification) 
    print()



main()