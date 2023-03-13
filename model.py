from process_data import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from sklearn.ensemble import RandomForestClassifier

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
    attention = Dense(N_features, activation='tanh')(inputs)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(N_features)(attention)

    # apply attention to inputs
    attention_output = Dot(axes=1)([inputs, attention])
    attention_output = Dense(N_features, activation='relu')(attention_output)

    # concatenate attention output with inputs
    merged = Concatenate()([attention_output, inputs])

    # define output layer
    output = Dense(N_labels, activation='sigmoid')(merged)

    # compile model
    model = Model(inputs=[inputs], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    plot_model(model, to_file='attention_model.png', show_shapes=True, show_layer_names=True)

    return model


def Define_Random_Forest_Model(n_estimator=50, max_depth=50):
    RF_Classifier = RandomForestClassifier(n_estimators=n_estimator, max_depth=max_depth)
    return RF_Classifier