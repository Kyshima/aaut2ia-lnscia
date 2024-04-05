import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import model_from_json

# Carregar dados do arquivo CSV
data = pd.read_csv("C:/Users/Patricia/Documents/GitHub/aaut2ia-lnscia/MachineLearning/Dataset-Preparation/DatasetBinary128.csv")

# Carregar e deserializar os dados de pixel
def load_pickle_data(row):
    return pickle.loads(eval(row['Informacao de Pixels']))

data['Informacao de Pixels'] = data.apply(load_pickle_data, axis=1)

# Dividir dados em recursos (features) e rótulos (labels)
X = np.array(data["Informacao de Pixels"])  # Recursos
y_crop = np.array(data["illness"])  # Rótulo "illness"

# Normalizar os dados de entrada
X = X / 255.0

# Redimensionar os dados de entrada para corresponder à entrada da CNN
X_resized = np.array([np.reshape(sample, (128, 128, 3)) for sample in X])

# Carregar a arquitetura do modelo
json_file = open('model_architecture.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Carregar os pesos do modelo treinado
model.load_weights("model_checkpoint.weights.h5")

# Realizar previsões
predictions = model.predict(X_resized)
predictions = (predictions > 0.5)  # Converte as probabilidades em previsões binárias

# Calcular métricas de avaliação
accuracy = accuracy_score(y_crop, predictions)
#precision = precision_score(y_crop, predictions, average='binary')
#recall = recall_score(y_crop, predictions, average='binary')
#f1 = f1_score(y_crop, predictions, average='binary')

# Imprimir métricas de avaliação
print("Accuracy:", accuracy)
#print("Precision:", precision)
#print("Recall:", recall)
#print("F1 Score:", f1)
