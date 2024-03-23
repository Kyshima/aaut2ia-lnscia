import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.python.keras import layers, models, callbacks
from PIL import Image
import pickle

def decode_image(pickled_data):
    image_array = pickle.loads(eval(pickled_data))
    image = Image.fromarray(image_array)
    return np.array(image)

# Load dataset
dataset = pd.read_csv("DatasetBinary128.csv")

# Decode binary image data and reshape
X = np.array([decode_image(data) for data in dataset['Informacao de Pixels']])
X = X.reshape(X.shape[0], 128, 128, 3)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(dataset['crop'])

# Split dataset
train_images, test_images, train_labels, test_labels = train_test_split(X, y, test_size=0.3, random_state=42)

# Define CNN model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(train_images, train_labels, epochs=20, batch_size = 100)

# Evaluate model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Make predictions
predictions = model.predict(test_images)
print("Sample prediction:", np.argmax(predictions[0]))
print("True label:", test_labels[0])