from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from PIL import Image
import pickle
import random

#Não se esqueçam de ver a cena dos callbacks

def decode_image(pickled_data):
    image_array = pickle.loads(eval(pickled_data))
    image = Image.fromarray(image_array)
    return np.array(image)

random.seed(100)

# Load dataset
dataset = pd.read_csv("DatasetBinary128.csv")

# Decode binary image data and reshape
X = np.array([decode_image(data) for data in dataset['Informacao de Pixels']])
X = X.reshape(X.shape[0], 128, 128, 3)

# Encode labels
label_encoder = LabelEncoder()
#Mudar consoante feature
y = label_encoder.fit_transform(dataset['crop'])

# Split dataset
train_images, test_images, train_labels, test_labels = train_test_split(X, y, test_size=0.3, random_state=42)

# Define CNN model
#Não ter medo de mudar os modelos porque cada coisa irá ter o seu próprio modelo
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(15, activation='softmax')
])

#Rede completamente convulucional do diogo para testar se é melhor
'''keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),
    keras.layers.MaxPooling2D((2, 2), padding='same'),
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2), padding='same'),
    keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    keras.layers.UpSampling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.UpSampling2D((2, 2)),
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')'''

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(train_images, train_labels, epochs=20, batch_size = 100)

# Evaluate model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nAccuracy:', test_acc)

# Make predictions
predictions = model.predict(test_images)

# Convert predictions to binary labels
binary_predictions = np.argmax(predictions, axis=1)

# Calculate precision, recall, and F1-score
precision = precision_score(test_labels, binary_predictions, average='weighted')
recall = recall_score(test_labels, binary_predictions, average='weighted')
f1 = f1_score(test_labels, binary_predictions, average='weighted')

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

print("Sample prediction:", binary_predictions[0])
print("True label:", test_labels[0])