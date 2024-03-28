import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder



df = pd.read_csv("NaturalLanguage/intreperterModel/new.csv", sep=";")

max_words = 1000  # Maximum number of words to keep
max_len = 100  # Maximum length of sequences

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['treatment'])
sequences = tokenizer.texts_to_sequences(df['treatment'])

X = pad_sequences(sequences, maxlen=max_len)

y = df['disease']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

model = Sequential([
    Embedding(max_words, 50),
    LSTM(64),
    Dense(len(df['disease']), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X, y_encoded, epochs=10, batch_size=32)

def generate_description(illness):
    description = ""
    # Tokenize and pad the input
    sequence = tokenizer.texts_to_sequences([illness])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    # Generate text
    predicted_index = np.argmax(model.predict(padded_sequence), axis=-1)[0]
    description = df.iloc[predicted_index]['treatment']
    return description

generated_description = generate_description('Brown Spot')
print(generated_description)