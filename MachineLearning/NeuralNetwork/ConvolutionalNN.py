import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.python.keras import callbacks
from PIL import Image
import pickle
from tensorflow import keras


def decode_image(pickled_data):
    image_array = pickle.loads(eval(pickled_data))

    image = Image.fromarray(image_array)
    return np.array(image)

dataset = pd.read_csv("DatasetBinary128.csv")

X = np.array([decode_image(data) for data in dataset['Informacao de Pixels']])
X = X.reshape(X.shape[0], 128, 128, 3)
y = dataset['crop']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the CNN model
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

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define callbacks
'''callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=5),
    callbacks.ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)
]'''

# Train the model
#model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
model.fit(X_train, y_train, epochs=20, batch_size = 32)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

accuracy = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes, average='weighted')
recall = recall_score(y_test, y_pred_classes, average='weighted')
f1 = f1_score(y_test, y_pred_classes, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Save the trained model
#model.save('crop_prediction_model.h5')
