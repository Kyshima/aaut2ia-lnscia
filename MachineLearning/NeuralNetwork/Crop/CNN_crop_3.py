import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from PIL import Image
import pickle
import random
from keras_tuner.tuners import RandomSearch
import matplotlib.pyplot as plt

random.seed(100)

dataset = pd.read_csv("C:/Users/Diana/Documents/aaut2ia-lnscia/MachineLearning/DatasetBinary128.csv")

def decode_image(pickled_data):
    image_array = pickle.loads(eval(pickled_data))
    image = Image.fromarray(image_array)
    return np.array(image)

X = np.array([decode_image(data) for data in dataset['Informacao de Pixels']])
X = X.reshape(X.shape[0], 128, 128, 3)
X = X / 255.00

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(dataset['crop'])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(hp.Int('conv1_units', min_value=16, max_value=64, step=16), (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(hp.Int('conv2_units', min_value=16, max_value=64, step=16), (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(hp.Int('conv3_units', min_value=16, max_value=64, step=16), (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(hp.Int('conv4_units', min_value=16, max_value=64, step=16), (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(hp.Int('conv5_units', min_value=16, max_value=64, step=16), (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(hp.Int('dense1_units', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(hp.Int('dense2_units', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(4, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

Model_Checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'C:/Users/Diana/Documents/aaut2ia-lnscia/MachineLearning/NeuralNetwork/Crop/cnn_crop_classification3.keras',
    monitor='val_loss', 
    save_best_only='True',
    verbose=1
)

Early_Stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5,
    verbose=1,
    restore_best_weights='True',
    min_delta=0.1
)

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,  # Number of hyperparameter combinations to try
    directory='C:/Users/Diana/Documents/aaut2ia-lnscia/MachineLearning/NeuralNetwork/Crop/HyperparametersTests/cnn_crop_classification3_hyper',  # Directory to store the tuning results
    project_name='cnn_hyperparameter_tuning'  # Name of the tuning project
)

tuner.search(x_train, y_train,
             epochs=10,
             validation_data=(x_test, y_test),
             callbacks=[Model_Checkpoint, Early_Stopping])

best_model = tuner.get_best_models(num_models=1)[0]

best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Best Hyperparameters:")
print(best_hyperparameters.values)

test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=2)
print('\nAccuracy:', test_acc)

predictions = best_model.predict(x_test)
binary_predictions = np.argmax(predictions, axis=1)

precision = precision_score(y_test, binary_predictions, average='weighted')
recall = recall_score(y_test, binary_predictions, average='weighted')
f1 = f1_score(y_test, binary_predictions, average='weighted')

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')