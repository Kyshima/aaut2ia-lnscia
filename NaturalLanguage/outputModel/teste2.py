import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import LambdaCallback
import sys
from utils import *

pd.set_option('display.max_colwidth', None)
df = pd.read_csv("new.csv", sep=";")

testOneDisease = df[df['formality'] == 'formal'].apply(lambda row: row['crop'] + re.compile(' ').sub('', row['disease']) + ' ' + row['treatment'], axis=1).apply(text_prepare) + " EOF"

def expand_description(df):
    expanded_data = []
    for index, row in df.iterrows():
        words = row['description'].split()
        leading_words = ''
        for i in range(len(words) - 1):
            leading_words += ' ' + words[i]
            expanded_data.append([leading_words.strip(), words[i+1]])
    expanded_df = pd.DataFrame(expanded_data, columns=['Leading Words', 'Next Word'])
    return expanded_df

expanded_df = expand_description(testOneDisease.to_frame(name="description"))

max_words = 1000  # Maximum number of words to keep
max_len = 100  # Maximum length of sequences

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(expanded_df['Leading Words'])
sequences = tokenizer.texts_to_sequences(expanded_df['Leading Words'])

X = pad_sequences(sequences, maxlen=max_len)

y = expanded_df['Next Word']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

model = Sequential([
    Embedding(max_words, 50),
    LSTM(128),
    Dense(len(expanded_df['Next Word']), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def on_epoch_end(model):
        print()
        print('----- Generating text after Epoch')
        #start_index = random.randint(0, len(testOneDisease) - maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('----- diversity:', diversity)

            generated = ''
            sentence = "wheatyellowrust"
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            while True or len(generated) < 100:
                sequence = tokenizer.texts_to_sequences([generated])
                padded_sequence = pad_sequences(sequence, maxlen=max_len)
                # Generate text
                predicted_index = np.argmax(model.predict(padded_sequence), axis=-1)[0]
                # Reverse encoding to get the actual disease label
                predicted_disease = label_encoder.inverse_transform([predicted_index])[0]

                generated += " " + predicted_disease
                print(generated)
                if predicted_disease == "EOF":
                    break


model.fit(X, y_encoded, epochs=200, batch_size=128)



on_epoch_end(model)

model.save("text-generator-model.keras")

with open("../../Final-Models/models/text-generator-tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("../../Final-Models/models/text-generator-label-encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

'''
df = pd.read_csv("NaturalLanguage/outputModel/preprocessed_dataset.csv")

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
    # Tokenize and pad the input
    sequence = tokenizer.texts_to_sequences([illness])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    # Generate text
    predicted_index = np.argmax(model.predict(padded_sequence), axis=-1)[0]
    # Reverse encoding to get the actual disease label
    predicted_disease = label_encoder.inverse_transform([predicted_index])[0]
    return predicted_disease

generated_description = generate_description('Brown Spot')
print(generated_description)
'''