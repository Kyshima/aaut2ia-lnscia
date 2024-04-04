from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
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
X = X / 255.00

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(dataset['crop'])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

inputs = keras.layers.Input((128, 128, 3))

# Encoder
'''conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)

conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)

conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

# Decoder (upsampling) path with skip connections
up4 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv3)
concat4 = keras.layers.concatenate([conv2, up4], axis=3)
print(inputs)
conv4 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat4)
conv4 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

up5 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv4)
concat5 = keras.layers.concatenate([conv1, up5], axis=3)
conv5 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(concat5)
conv5 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

outputs = keras.layers.Conv2D(4, (1, 1), activation='softmax', padding='valid')(conv5)

model = keras.Model(inputs=inputs, outputs=outputs)'''

'''
inputs = keras.layers.Input(shape=(128, 128, 3))
x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.MaxPooling2D((2, 2))(x)

x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.MaxPooling2D((2, 2))(x)

x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.MaxPooling2D((2, 2))(x)

# Uppling
x = keras.layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.UpSampling2D((2, 2))(x)

x = keras.layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.UpSampling2D((2, 2))(x)

x = keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.UpSampling2D((2, 2))(x)

# Output layer
outputs = keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)

# Create the model
model = keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()
'''
dilation_rate = (1,1)
activation = 'relu'
reg = 0.01
dropout = 0
    
c1 =  keras.layers.Conv2D(32, 3 , activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (inputs)
c1 =  keras.layers.Dropout(dropout) (c1)
c1 =  keras.layers.Conv2D(32, 3 , activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (c1)
p1 =  keras.layers.MaxPooling2D((2, 2)) (c1)

c2 =  keras.layers.Conv2D(64, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (p1)
c2 =  keras.layers.Dropout(dropout) (c2)
c2 =  keras.layers.Conv2D(64, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (c2)
p2 =  keras.layers.MaxPooling2D((2, 2)) (c2)

c3 =  keras.layers.Conv2D(128, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (p2)
c3 =  keras.layers.Dropout(dropout) (c3)
c3 =  keras.layers.Conv2D(128, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (c3)
p3 =  keras.layers.MaxPooling2D((2, 2)) (c3)
# p3 = BatchNormalization()(p3)

c4 =  keras.layers.Conv2D(256, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (p3)
c4 =  keras.layers.Dropout(dropout) (c4)
c4 =  keras.layers.Conv2D(256, 3, activation=activation, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (c4)
p4 =  keras.layers.MaxPooling2D(pool_size=(2, 2)) (c4)
# p4 = BatchNormalization()(p4)

c5 =  keras.layers.Conv2D(512, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (p4)
c5 =  keras.layers.Dropout(dropout) (c5)
c5 =  keras.layers.Conv2D(512, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (c5)

#EXPANSIVE Path (DECODER)
    
u6 =  keras.layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same') (c5)
u6 =  keras.layers.concatenate([u6, c4])
c6 =  keras.layers.Conv2D(256, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (u6)
c6 =  keras.layers.Dropout(dropout) (c6)
c6 =  keras.layers.Conv2D(256, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (c6)
    
u7 =  keras.layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same') (c6)
u7 =  keras.layers.concatenate([u7, c3])
c7 =  keras.layers.Conv2D(128, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (u7)
c7 =  keras.layers.Dropout(dropout) (c7)
c7 =  keras.layers.Conv2D(128, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (c7)
    
u8 =  keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same') (c7)
u8 =  keras.layers.concatenate([u8, c2])
c8 =  keras.layers.Conv2D(64, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (u8)
c8 =  keras.layers.Dropout(dropout) (c8)
c8 =  keras.layers.Conv2D(64, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (c8)

u9 =  keras.layers.Conv2DTranspose(32, 2, strides=(2, 2), padding='same') (c8)
u9 =  keras.layers.concatenate([u9, c1], axis=3)
c9 =  keras.layers.Conv2D(32, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (u9)
c9 =  keras.layers.Dropout(dropout) (c9)
c9 =  keras.layers.Conv2D(32, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (c9)
    
outputs = keras.layers.Conv2D(4, 1, activation='softmax') (c9) 

model = keras.Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, batch_size=100)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nAccuracy:', test_acc)

predictions = model.predict(x_test)

binary_predictions = (predictions > 0.5).astype(np.uint8)

precision = precision_score(y_test.flatten(), binary_predictions.flatten())
recall = recall_score(y_test.flatten(), binary_predictions.flatten())
f1 = f1_score(y_test.flatten(), binary_predictions.flatten())

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')