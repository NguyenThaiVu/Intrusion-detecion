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
    print ("Model Accuracy:" "\n", accuracy)
    print()
    print("Confusion matrix:" "\n", confusion_matrix)
    print()
    print("Classification report:" "\n", classification) 
    print()



def Evaluate_Report(Y_test, Y_pred):
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