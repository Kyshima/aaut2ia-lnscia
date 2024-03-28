import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding  # Importing Embedding layer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


df = pd.read_csv("NaturalLanguage/intreperterModel/new.csv", sep=";")

# Sample text data
text_data = """
This is a sample text data for training a simple RNN-based text generator. 
You can replace it with your own text data for better results.
"""

X_text = df['treatment']

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_text)
total_words = len(tokenizer.word_index) + 1

# Creating input sequences
input_sequences = []
for line in X_text:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

# Padding sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Creating predictors and labels
predictors, label = input_sequences[:, :-1], input_sequences[:, -1]

# One-hot encoding labels
label = tf.keras.utils.to_categorical(label, num_classes=total_words)

# Building the model
model = Sequential()
model.add(Embedding(total_words, 100))
model.add(LSTM(150, return_sequences=True))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model
model.fit(predictors, label, epochs=20, verbose=1)


# Function to generate text
def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = model.predict_step(token_list)

        # Instead of direct comparison, we need to find the index of the maximum value
        predicted_index = np.argmax(predicted)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        seed_text = output_word
    return seed_text


# Example usage
generated_text = generate_text("Hit 'em with", max_sequence_len, model, max_sequence_len)
print(generated_text)