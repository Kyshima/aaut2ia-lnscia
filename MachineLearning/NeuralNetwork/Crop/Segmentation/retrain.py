import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

print('Estou a carregarzinho o csvzinho...')

# Carregar dados do arquivo CSV
data = pd.read_csv("C:/Users/Patricia/Documents/GitHub/aaut2ia-lnscia/MachineLearning/Dataset-Preparation/DatasetBinary128.csv")

# Carregar e deserializar os dados de pixel
def load_pickle_data(row):
    return pickle.loads(eval(row['Informacao de Pixels']))

data['Informacao de Pixels'] = data.apply(load_pickle_data, axis=1)

# Dividir dados em recursos (features) e rótulos (labels)
X = np.array(data["Informacao de Pixels"])  # Recursos
y_crop = np.array(data["crop"])  # Rótulo "crop"

# Dividir dados em conjuntos de treinamento e teste (70% treinamento, 30% teste)
X_train, X_test, y_crop_train, y_crop_test = train_test_split(
    X, y_crop, test_size=0.3, random_state=42
)

# Normalizar os dados de entrada
X_train = X_train / 255.0
X_test = X_test / 255.0

# Redimensionar os dados de entrada para corresponder à entrada da CNN
X_train_resized = np.array([np.reshape(sample, (128, 128, 3)) for sample in X_train])
X_test_resized = np.array([np.reshape(sample, (128, 128, 3)) for sample in X_test])

# Carregar a arquitetura do modelo a partir do arquivo JSON
with open("model_architecture.json", "r") as json_file:
    model_json = json_file.read()

# Construir o modelo a partir da arquitetura carregada
model = models.model_from_json(model_json)

# Carregar os pesos do modelo
model.load_weights("model_checkpoint.weights.h5")

# Compilar o modelo
model.compile(
              #optimizer='adamW',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Salvar a arquitetura do modelo (apenas uma vez)
with open("model_architecture.json", "w") as json_file:
    json_file.write(model_json)

# Definir o caminho para salvar os pesos do modelo
checkpoint_path = "model_checkpoint.weights.h5"

# Definir o callback para salvar os pesos do modelo com base na perda de validação ('val_loss')
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path,
                                      save_weights_only=True,
                                      save_best_only=True,
                                      monitor='val_loss',  # Monitorar a perda de validação
                                      mode='min',          # Salvar o modelo quando a perda de validação diminuir
                                      verbose=1)

# Retomar o treinamento do modelo
model.fit(X_train_resized, y_crop_train,
          epochs=10,
          batch_size=128,
          verbose=1,
          validation_data=(X_test_resized, y_crop_test),
          callbacks=[checkpoint_callback])
