from process_data import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Concatenate, Dot, Flatten
from keras.layers import LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, TimeDistributed, RepeatVector, Permute
from keras.models import Model
from keras.utils import plot_model
from keras.models import save_model


def Define_Attention_Model(N_features, N_labels):
    # define input shape
    inputs = Input(shape=(N_features,))

    # define attention layer
    attention = Dense(N_features, activation='relu')(inputs)
    attention = Dense(N_features, activation='softmax')(attention)
    # attention = RepeatVector(N_features)(attention)

    # apply attention to inputs
    attention_output = Dot(axes=1)([inputs, attention])
    attention_output = Dense(N_features, activation='relu')(attention_output)

    # concatenate attention output with inputs
    merged = Concatenate()([attention_output, inputs])

    # MLP
    mlp = Dense(N_features, activation='relu')(merged)

    # define output layer
    output = Dense(N_labels, activation='sigmoid')(mlp)

    # compile model
    model = Model(inputs=[inputs], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    plot_model(model, to_file='attention_model.png', show_shapes=True, show_layer_names=True)

    return model


def Define_Random_Forest_Model(n_estimator=50, max_depth=50):
    RF_Classifier = RandomForestClassifier(n_estimators=n_estimator, max_depth=max_depth)
    return RF_Classifier


def Define_Decision_Tree(max_depth=10):
    tree_clf = DecisionTreeClassifier(max_depth=max_depth)
    return tree_clf

def Define_Logistic_Regression():
    lr = LogisticRegression()
    return lr

def Define_KNN(num_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=num_neighbors)
    return knn

def Define_MLP(hidden_size=(32, 32), max_iter=100):
    mlp = MLPClassifier(hidden_layer_sizes=hidden_size, max_iter=max_iter)
    return mlp