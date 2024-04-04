import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from PIL import Image
import pickle
import random
from keras_tuner.tuners import RandomSearch

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

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def build_model(hp):
    input_shape = (128, 128, 3)  # Input shape of your images
    num_classes = 4  # Number of classes (crop types) to predict

    # Backbone CNN (VGG16)
    backbone = tf.keras.applications.VGG16(input_shape=input_shape, include_top=False)
    backbone.trainable = False

    # Additional layers for object detection
    x = keras.layers.Flatten()(backbone.output)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(4*num_classes, activation='sigmoid')(x)  # 4*num_classes for bounding box coordinates and class probabilities
    outputs = keras.layers.Reshape((num_classes, 4))(x)  # Reshape to separate bounding box coordinates and class probabilities

    # Define the model
    model = keras.models.Model(inputs=backbone.input, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='mse',  # Mean squared error loss for bounding box regression
                  metrics=['accuracy'])

    return model

Model_Checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'C:/Users/Diana/Documents/aaut2ia-lnscia/MachineLearning/NeuralNetwork/Crop/cnn_crop_detection.keras',
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
    directory='C:/Users/Diana/Documents/aaut2ia-lnscia/MachineLearning/NeuralNetwork/Crop/HyperparametersTests/cnn_crop_detection_hyper',  # Directory to store the tuning results
    project_name='crop_hyperparameter_tuning'  # Name of the tuning project
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
binary_predictions = (predictions > 0.5).astype("int32")

precision = precision_score(y_test, binary_predictions, average='weighted')
recall = recall_score(y_test, binary_predictions, average='weighted')
f1 = f1_score(y_test, binary_predictions, average='weighted')

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
