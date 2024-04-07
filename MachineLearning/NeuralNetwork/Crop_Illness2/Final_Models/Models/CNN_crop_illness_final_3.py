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

model = keras.Sequential()

# Convolutional Layer 1
model.add(keras.layers.Conv2D(48, (3, 3), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 3)))

# Convolutional Layer 2
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Convolutional Layer 3
model.add(keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 3)))

# Convolutional Layer 4
model.add(keras.layers.Conv2D(32, (7, 7), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 2)))

# Convolutional Layer 5
model.add(keras.layers.Conv2D(48, (7, 7), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))

model.add(keras.layers.Flatten())

# Dense Layer 1
model.add(keras.layers.Dense(288, activation='relu'))
model.add(keras.layers.Dropout(0.3))

# Dense Layer 2
model.add(keras.layers.Dense(448, activation='relu'))
model.add(keras.layers.Dropout(0.3))

# Dense Layer 3
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.3))

# Output Layer
model.add(keras.layers.Dense(14, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


Model_Checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'C:/Users/Diana/Documents/aaut2ia-lnscia/MachineLearning/NeuralNetwork/Crop_Illness2/Final_Models/Models_Exported/cnn_crop_illness3.keras',
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

# Fit the model and save training history
history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), callbacks=[Model_Checkpoint, Early_Stopping])

# Save the training history to a file
with open('training_history3_crop_illness.pkl', 'wb') as file:
    pickle.dump(history.history, file)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nAccuracy:', test_acc)

predictions = model.predict(x_test)
binary_predictions = np.argmax(predictions, axis=1)

precision = precision_score(y_test, binary_predictions, average='weighted')
recall = recall_score(y_test, binary_predictions, average='weighted')
f1 = f1_score(y_test, binary_predictions, average='weighted')

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')