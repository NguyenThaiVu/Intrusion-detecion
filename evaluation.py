from sklearn import metrics
from sklearn.model_selection import cross_val_score
import numpy as np

from process_data import *


def Evaluate_Model(model, X_test, y_test, is_binary_classifier=False):

    y_pred =  model.predict(X_test)
    
    if is_binary_classifier == False:
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)
    else:
        y_pred = (y_pred > 0.5)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    classification = metrics.classification_report(y_test, y_pred)
    print()
    print('============================== Model Evaluation ==============================')
    print()
    print ("Model Accuracy:" "\n", accuracy)
    print()
    print("Confusion matrix:" "\n", confusion_matrix)
    print()
    print("Classification report:" "\n", classification) 
    print()


def Evaluate_Model_DDos_Attack(model, except_attack):

    # Process data
    if "index" in except_attack.columns:
        except_attack = except_attack.drop(['index'],axis=1)

    y_ddos = except_attack[' Label'].copy()
    X_ddos = except_attack.drop([' Label'],axis=1)

    # Scale numerical features to have zero mean and unit variance  
    scaler = StandardScaler()
    X_ddos = scaler.fit_transform(X_ddos.select_dtypes(include=['float32','float16','int32','int16','int8']))

    # Predict
    y_pred =  model.predict(X_ddos)
    y_pred = (y_pred > 0.5)

    accuracy = metrics.accuracy_score(y_ddos, y_pred)
    confusion_matrix = metrics.confusion_matrix(y_ddos, y_pred)
    classification = metrics.classification_report(y_ddos, y_pred)
    print()
    print('============================== Model Evaluation on excetp type ==============================')
    print()
    print ("Model Accuracy:" "\n", accuracy)
    print()
    print("Confusion matrix:" "\n", confusion_matrix)
    print()
    print("Classification report:" "\n", classification) 
    print()