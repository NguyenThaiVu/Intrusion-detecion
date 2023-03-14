import tensorflow as tf
from tensorflow import keras
from kerastuner.tuners import GridSearch

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize the pixel values to be between 0 and 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Define your deep neural network model using Keras
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))

    model.add(keras.layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(keras.layers.Dense(units=10, activation='softmax'))
    
    model.compile(optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create an instance of the GridSearch tuner
tuner = GridSearch(build_model, objective='val_accuracy', max_trials=10, directory='my_dir', project_name='my_project')

# Start the search
tuner.search(x_train, y_train, epochs=1, validation_data=(x_val, y_val))

# Retrieve the best hyperparameters
best_hyperparams = tuner.get_best_hyperparameters()[0]
