import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image
import pickle
import random
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.environ['PYTHONHASHSEED'] = '0'

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
y = label_encoder.fit_transform(dataset['crop'])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed)

model = keras.Sequential()
model.add(keras.layers.Input(shape=(128, 128, 3)))

# Convolutional Layer 1
model.add(keras.layers.Conv2D(48, (3, 3), activation='relu', padding='same'))

# MaxPooling Layer 1
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Convolutional Layer 2
model.add(keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'))

# MaxPooling Layer 2
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Convolutional Layer 3
model.add(keras.layers.Conv2D(16, (7, 7), activation='relu', padding='same'))

# MaxPooling Layer 3
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

#Flatten Layer
model.add(keras.layers.Flatten())

# Output Layer
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(4, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

Early_Stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=20,
    verbose=1,
    restore_best_weights='True',
    min_delta=0.01
)

# Fit the model and save training history
history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), callbacks=[Early_Stopping])

#Save the training history to a file
with open('C:/Users/Diana/Documents/aaut2ia-lnscia/MachineLearning/NeuralNetwork/Crop/Final_Models/Graphs/training_history1_crop.pkl', 'wb') as file:
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

# Save the trained model
model.save('C:/Users/Diana/Documents/aaut2ia-lnscia/MachineLearning/NeuralNetwork/Crop/Final_Models/Models_Exported/crop1.h5')

#-----------------------------Graphs---------------------------------

# Confusion matrix
conf_matrix = confusion_matrix(y_test, binary_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Plot the graph of accuracy versus epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()
plt.show()

# Plot the graph of loss versus epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.show()


