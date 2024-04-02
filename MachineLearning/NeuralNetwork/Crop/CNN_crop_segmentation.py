from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from PIL import Image
import pickle
import random

def decode_image(pickled_data):
    image_array = pickle.loads(eval(pickled_data))
    image = Image.fromarray(image_array)
    return np.array(image)

random.seed(100)

dataset = pd.read_csv("C:/Users/Diana/Documents/aaut2ia-lnscia/DatasetBinary128.csv")

X = np.array([decode_image(data) for data in dataset['Informacao de Pixels']])
X = X / 255.0 
X = X.reshape(X.shape[0], 128, 128, 3)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(dataset['crop'])

train_images, test_images, train_labels, test_labels = train_test_split(X, y, test_size=0.3, random_state=42)

train_masks = np.expand_dims(train_labels, axis=-1)
test_masks = np.expand_dims(test_labels, axis=-1)

inputs = keras.layers.Input((128, 128, 3))

# Encoder
conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

# Decoder
conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
up1 = keras.layers.UpSampling2D((2, 2))(conv3)

merge1 = keras.layers.concatenate([conv2, up1], axis=-1)
conv4 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(merge1)
conv4 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
up2 = keras.layers.UpSampling2D((2, 2))(conv4)

merge2 = keras.layers.concatenate([conv1, up2], axis=-1)
conv5 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(merge2)
conv5 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

outputs = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(conv5)

model = keras.Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_masks, epochs=20, batch_size=100)

test_loss, test_acc = model.evaluate(test_images, test_masks, verbose=2)
print('\nAccuracy:', test_acc)

predictions = model.predict(test_images)

binary_predictions = (predictions > 0.5).astype(np.uint8)

precision = precision_score(test_masks.flatten(), binary_predictions.flatten())
recall = recall_score(test_masks.flatten(), binary_predictions.flatten())
f1 = f1_score(test_masks.flatten(), binary_predictions.flatten())

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

print("Sample prediction:", binary_predictions[0])
print("True mask:", test_masks[0])
