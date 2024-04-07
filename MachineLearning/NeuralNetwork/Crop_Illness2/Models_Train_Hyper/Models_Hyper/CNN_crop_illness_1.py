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

random_seed = 100
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

dataset = pd.read_csv("C:/Users/Diana/Documents/aaut2ia-lnscia/MachineLearning/DatasetBinary128.csv")

def decode_image(pickled_data):
    image_array = pickle.loads(eval(pickled_data))
    image = Image.fromarray(image_array)
    return np.array(image)

X = np.array([decode_image(data) for data in dataset['Informacao de Pixels']])
X = X.reshape(X.shape[0], 128, 128, 3)
X = X / 255.00

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(dataset['crop_illness'])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Hyperparameters
min_filters = 16  # Minimum number of filters for convolutional layers 
max_filters = 64  # Maximum number of filters for convolutional layers
filter_step = 16  # Step size for increasing the number of filters in convolutional layers
kernel_sizes = [3, 5, 7]  # List of kernel sizes for convolutional layers 
activation_function = "relu"  # Activation function used in convolutional and dense layers
pool_sizes = [2, 3]  # List of pool sizes for max-pooling layers
dense_min_units = 32  # Minimum number of units for dense (fully connected) layers
dense_max_units = 128  # Maximum number of units for dense (fully connected) layers
dense_step_units = 32  # Step size for increasing the number of units in dense (fully connected) layers

def build_model(hp):
    model = keras.Sequential()

    # Convolutional Layer 1
    conv1_units = hp.Int('conv1_units', min_value=min_filters, max_value=max_filters, step=filter_step)
    kernel_size1 = hp.Choice('kernel_size1', values=kernel_sizes)    
    model.add(keras.layers.Conv2D(conv1_units, (kernel_size1, kernel_size1), activation=activation_function, padding='same'))
   
    # MaxPooling Layer 1
    pool_size1 = (hp.Choice('pool_size1_height', values=pool_sizes), hp.Choice('pool_size1_width', values=pool_sizes))
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size1))
    
    # Convolutional Layer 2
    conv2_units = hp.Int('conv2_units', min_value=min_filters, max_value=max_filters, step=filter_step)
    kernel_size2 = hp.Choice('kernel_size2', values=kernel_sizes)
    model.add(keras.layers.Conv2D(conv2_units, (kernel_size2, kernel_size2), activation=activation_function, padding='same'))
    
    # MaxPooling Layer 2
    pool_size2 = (hp.Choice('pool_size2_height', values=pool_sizes), hp.Choice('pool_size2_width', values=pool_sizes))
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size2))
    
    # Convolutional Layer 3
    conv3_units = hp.Int('conv3_units', min_value=min_filters, max_value=max_filters, step=filter_step)
    kernel_size3 = hp.Choice('kernel_size3', values=kernel_sizes)    
    model.add(keras.layers.Conv2D(conv3_units, (kernel_size3, kernel_size3), activation=activation_function, padding='same'))
    
    # Flatten Layer
    model.add(keras.layers.Flatten())
    
    # Output Layer
    model.add(keras.layers.Dense(hp.Int('dense_units', min_value = dense_min_units, max_value = dense_max_units, step = dense_step_units), activation='relu'))
    model.add(keras.layers.Dense(14, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

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
    directory='C:/Users/Diana/Documents/aaut2ia-lnscia/MachineLearning/NeuralNetwork/Crop_Illness2/Models_Train_Hyper/HyperparametersTests/cnn_crop_illness1_hyper',  # Directory to store the tuning results
    project_name='cnn_hyperparameter_tuning'  # Name of the tuning project
)

tuner.search(x_train, y_train,
             epochs=10,
             validation_data=(x_test, y_test),
             callbacks=[Early_Stopping])

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