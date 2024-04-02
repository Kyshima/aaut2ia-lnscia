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
import keras_tuner
from kerastuner.tuners import RandomSearch
from kerastuner import HyperParameters

random.seed(100)

dataset = pd.read_csv("C:/Users/Diana/Documents/aaut2ia-lnscia/MachineLearning/DatasetBinary128.csv")

def decode_image(pickled_data):
    image_array = pickle.loads(eval(pickled_data))
    image = Image.fromarray(image_array)
    return np.array(image)

X = np.array([decode_image(data) for data in dataset['Informacao de Pixels']])
X = X.reshape(X.shape[0], 128, 128, 3)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(dataset['crop'])

train_images, test_images, train_labels, test_labels = train_test_split(X, y, test_size=0.3, random_state=42)

def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(hp.Int('conv1_units', min_value=16, max_value=64, step=16), (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(hp.Int('conv2_units', min_value=16, max_value=64, step=16), (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(hp.Int('conv3_units', min_value=16, max_value=64, step=16), (3, 3), activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(hp.Int('dense_units', min_value=32, max_value=128, step=32), activation='relu'))
    model.add(keras.layers.Dense(15, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

Model_Checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'C:/Users/Diana/Documents/aaut2ia-lnscia/MachineLearning/NeuralNetwork/Crop/cnn_crop_classification.keras',
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
    directory='C:/Users/didi2/Documents/aaut2ia-lnscia/MachineLearning/NeuralNetwork/Crop/HyperparametersTests/cnn_crop_classification_hyper',  # Directory to store the tuning results
    project_name='cnn_hyperparameter_tuning'  # Name of the tuning project
)

tuner.search(train_images, train_labels,
             epochs=10,
             validation_data=(test_images, test_labels),
             callbacks=[Model_Checkpoint, Early_Stopping])

best_model = tuner.get_best_models(num_models=1)[0]

best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Best Hyperparameters:")
print(best_hyperparameters.values)

test_loss, test_acc = best_model.evaluate(test_images, test_labels, verbose=2)
print('\nAccuracy:', test_acc)

predictions = best_model.predict(test_images)
binary_predictions = np.argmax(predictions, axis=1)

precision = precision_score(test_labels, binary_predictions, average='weighted')
recall = recall_score(test_labels, binary_predictions, average='weighted')
f1 = f1_score(test_labels, binary_predictions, average='weighted')

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
