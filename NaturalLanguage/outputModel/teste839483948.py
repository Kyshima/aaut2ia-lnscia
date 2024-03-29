import numpy as np
import random
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import LambdaCallback
import sys

df = pd.read_csv("NaturalLanguage/outputModel/preprocessed_dataset.csv")

# Extract text from the 'treatment' column
text = ' '.join(df['treatment'].astype(str))

# Create character-level mappings
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# Create training data
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=bool)
y = np.zeros((len(sentences), len(chars)), dtype=bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# Define the RNN model
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Function to sample the next character
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Function to generate text
def on_epoch_end(epoch, _):
    if epoch == 59:
        print()
        print('----- Generating text after Epoch: %d' % epoch)
        start_index = random.randint(0, len(text) - maxlen - 1)
        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        for temperature in [0.2, 0.5, 1.0, 1.2]:
            print('----- Temperature:', temperature)
            sys.stdout.write(generated)
            for i in range(400):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.
                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, temperature)
                next_char = indices_char[next_index]
                generated += next_char
                sentence = sentence[1:] + next_char
                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

# Callback to generate text after each epoch
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

# Train the model
model.fit(x, y, batch_size=128, epochs=60, callbacks=[print_callback])
