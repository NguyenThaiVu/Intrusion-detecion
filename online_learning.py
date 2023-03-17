import sys
sys.dont_write_bytecode = True

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[1], 'GPU')

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from process_data import *
from model import *
from evaluation import *

# --------------- Hyper parameter ---------------------
N_features = 78
N_labels = 1
# -----------------------------------------------------

def My_Evaluate(Y_test, Y_pred):
    accuracy = metrics.accuracy_score(Y_test, Y_pred)
    confusion_matrix = metrics.confusion_matrix(Y_test, Y_pred)
    classification = metrics.classification_report(Y_test, Y_pred)
    print()
    print ("Model Accuracy:" "\n", accuracy)
    print()
    print("Confusion matrix:" "\n", confusion_matrix)
    print()
    print("Classification report:" "\n", classification) 
    print()


def Process_Label_Just_DDos(data):

    # ['BENIGN' 'DDoS' 'PortScan' 'Bot' 'Infiltration' 'Web Attack � Brute Force' 
    # 'Web Attack � XSS' 'Web Attack � Sql Injection']
    attackType = data[' Label'].unique()    
    print(Counter(data[' Label']))

    if "index" in data.columns: data = data.drop(['index'],axis=1)

    # Get data just Bot
    ddos_data = data[(data[' Label'] == "Bot")]
    ddos_data = ddos_data.reset_index(drop=False)
    ddos_data[' Label'] = ddos_data[' Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    ddos_data[' Label'] = ddos_data[' Label'].astype(np.float32)

    if "index" in ddos_data.columns: ddos_data = ddos_data.drop(['index'],axis=1)
    print("Shape DDoS: {}".format(ddos_data.shape))

    # Get data include: DDoS and BENIGN
    data = data[(data[' Label'] == "DDoS") | (data[' Label'] == "BENIGN")]
    data = data.reset_index(drop=False)
    data[' Label'] = data[' Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    data[' Label'] = data[' Label'].astype(np.float32)
    
    if "index" in data.columns: data = data.drop(['index'],axis=1)

    print("Shape new data: {}".format(data.shape))


    return (data, ddos_data)

def main():

    df = pd.read_csv(r'data/intrusion_detection_dataset.csv')
    df = Remove_INF_and_NaN(df)

    (data, ddos_data) = Process_Label_Just_DDos(df)

    (X_train, X_test, Y_train, Y_test, scaler) = Prepare_Data_Binary(data, is_handle_imbalance=True, is_scale=True)

    (X_train, X_train_2, Y_train, Y_train_2) = train_test_split(X_train, Y_train, test_size=0.1)

    print('\n----------------------------------------')
    print("[INFO] Finish prepare data")
    print("Number of samples per class: {}".format(Counter(Y_train)))
    print("X_train: {}".format(X_train.shape))
    print("Y_train: {}".format(Y_train.shape))
    print('----------------------------------------\n')
    
    # Define model
    print("\n-------------------------- Start training --------------------------")
    # model = Define_Decision_Tree()
    # model.fit(X_train, Y_train)
    model = Define_Attention_Model(N_features, N_labels)
    model.fit(X_train, Y_train, epochs=2, batch_size=128)

    # Evaluation
    print("\n-------------------------- Start evaluation on test set ----------------------")
    print("Number of samples per class: {}".format(Counter(Y_test)))
    print("X_train: {}".format(X_test.shape))
    print("Y_train: {}".format(Y_test.shape))

    Y_pred =  model.predict(X_test)
    Y_pred = (Y_pred > 0.5) * 1.0

    My_Evaluate(Y_test, Y_pred)


    # Evaluation just Bot
    print("\n-------------------------- Start predict only Bot -------------------------- ")
    Y_test_ddos = ddos_data[' Label'].copy()
    X_test_ddos = ddos_data.drop([' Label'],axis=1)
    X_test_ddos = scaler.transform(X_test_ddos)

    print("Number of samples per class: {}".format(Counter(Y_test_ddos)))
    print("X_train: {}".format(X_test_ddos.shape))
    print("Y_train: {}".format(Y_test_ddos.shape))

    Y_pred_ddos =  model.predict(X_test_ddos)
    Y_pred_ddos = (Y_pred_ddos > 0.5) * 1.0

    My_Evaluate(Y_test_ddos, Y_pred_ddos)
    
    # Transfer learning on Bot
    print("\n-------------------------- Start transfer learning on Bot and Train_2 -------------------------- ")

    print("Number of samples per class: {}".format(Counter(Y_test_ddos)))
    print("X_train: {}".format(X_test_ddos.shape))
    print("Y_train: {}".format(Y_test_ddos.shape))
    X_train_2 = np.concatenate([X_test_ddos, X_train_2], axis=0)
    Y_train_2 = np.concatenate([Y_test_ddos, Y_train_2], axis=0)
    model.fit(X_train_2, Y_train_2, epochs=3, batch_size=64)


    # Evaluation just Bot
    print("\n-------------------------- Start predict on Bot and Train_2 -------------------------- ")
    print("Number of samples per class: {}".format(Counter(Y_test_ddos)))
    print("X_train: {}".format(X_test_ddos.shape))
    print("Y_train: {}".format(Y_test_ddos.shape))

    Y_pred_ddos =  model.predict(X_test_ddos)
    Y_pred_ddos = (Y_pred_ddos > 0.5) * 1.0

    My_Evaluate(Y_test_ddos, Y_pred_ddos)


    # Evaluation
    print("\n-------------------------- Start evaluation AGAIN on test set ----------------------")
    print("Number of samples per class: {}".format(Counter(Y_test)))
    print("X_train: {}".format(X_test.shape))
    print("Y_train: {}".format(Y_test.shape))

    Y_pred =  model.predict(X_test)
    Y_pred = (Y_pred > 0.5) * 1.0

    My_Evaluate(Y_test, Y_pred)



main()