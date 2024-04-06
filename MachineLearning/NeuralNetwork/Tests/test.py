from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import pickle
import random
from keras import regularizers 

def decode_image(pickled_data):
    image_array = pickle.loads(eval(pickled_data))
    image = Image.fromarray(image_array)
    return np.array(image)

random.seed(100)

dataset = pd.read_csv("C:/Users/didi2/Documents/aaut2ia-lnscia/MachineLearning/DatasetBinary128.csv")

X = np.array([decode_image(data) for data in dataset['Informacao de Pixels']])
X = X.reshape(X.shape[0], 128, 128, 3)
X = X / 255.0

label_encoder = LabelEncoder()
y = np.array([decode_image(data) for data in dataset['pixel_labels']])  # Assuming 'pixel_labels' contains segmentation masks

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

inputs = keras.layers.Input((128, 128, 3))
dilation_rate = (1, 1)
activation = 'relu'
reg = 0.01
dropout = 0.5

# Encoder  
c1 = keras.layers.Conv2D(32, 3 , activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg))(inputs)
c1 = keras.layers.Dropout(dropout)(c1)
c1 = keras.layers.Conv2D(32, 3 , activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg))(c1)
p1 = keras.layers.MaxPooling2D((2, 2))(c1)

c2 = keras.layers.Conv2D(64, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg))(p1)
c2 = keras.layers.Dropout(dropout)(c2)
c2 = keras.layers.Conv2D(64, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg))(c2)
p2 = keras.layers.MaxPooling2D((2, 2))(c2)

c3 = keras.layers.Conv2D(128, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg))(p2)
c3 = keras.layers.Dropout(dropout)(c3)
c3 = keras.layers.Conv2D(128, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg))(c3)
p3 = keras.layers.MaxPooling2D((2, 2))(c3)

c4 = keras.layers.Conv2D(256, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg))(p3)
c4 = keras.layers.Dropout(dropout)(c4)
c4 = keras.layers.Conv2D(256, 3, activation=activation, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg))(c4)
p4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

c5 = keras.layers.Conv2D(512, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg))(p4)
c5 = keras.layers.Dropout(dropout)(c5)
c5 = keras.layers.Conv2D(512, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg))(c5)

# Decoder
u6 = keras.layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(c5)
u6 = keras.layers.concatenate([u6, c4])
c6 = keras.layers.Conv2D(256, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg))(u6)
c6 = keras.layers.Dropout(dropout)(c6)
c6 = keras.layers.Conv2D(256, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg))(c6)

u7 = keras.layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(c6)
u7 = keras.layers.concatenate([u7, c3])
c7 = keras.layers.Conv2D(128, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg))(u7)
c7 = keras.layers.Dropout(dropout)(c7)
c7 = keras.layers.Conv2D(128, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg))(c7)

u8 = keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(c7)
u8 = keras.layers.concatenate([u8, c2])
c8 = keras.layers.Conv2D(64, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg))(u8)
c8 = keras.layers.Dropout(dropout)(c8)
c8 = keras.layers.Conv2D(64, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg))(c8)

u9 = keras.layers.Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(c8)
u9 = keras.layers.concatenate([u9, c1], axis=3)
c9 = keras.layers.Conv2D(32, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg))(u9)
c9 = keras.layers.Dropout(dropout)(c9)
c9 = keras.layers.Conv2D(32, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg))(c9)

# Output layer
outputs = keras.layers.Conv2D(1, 1, activation='sigmoid')(c9)  # 1 channel for segmentation mask

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Use binary cross-entropy for binary segmentation
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, batch_size=100)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nAccuracy:', test_acc)

predictions = model.predict(x_test)

# Convert predictions to binary masks
threshold = 0.5
binary_predictions = (predictions > threshold).astype(np.uint8)

# Compute evaluation metrics (e.g., IoU)
# IoU can be computed using libraries like TensorFlow or manually
# Here's a manual calculation of IoU for demonstration purposes
intersection = np.logical_and(y_test, binary_predictions)
union = np.logical_or(y_test, binary_predictions)
iou = np.sum(intersection) / np.sum(union)

print(f'IoU: {iou}')
