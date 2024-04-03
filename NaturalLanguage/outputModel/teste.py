import numpy as np
import random
import pandas as pd
import pickle
import re
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from keras.callbacks import LambdaCallback
import sys
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
import os
nltk.download('stopwords')
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

pd.set_option('display.max_colwidth', None)

'''--------------------------------- Intent recgonizer ---------------------------------'''
'''
df = pd.read_csv("NaturalLanguage/outputModel/new.csv", sep=";")
dialogue_df = pd.read_csv('NaturalLanguage/outputModel/data/dialogues.tsv', sep='\t').sample(1495, random_state=0)

dialogue_df.head()
dialogue_df['text'] = dialogue_df['text'].apply(text_prepare)
df['treatment'] = df['treatment'].apply(text_prepare)

X = np.concatenate([dialogue_df['text'].values, df['treatment'].values])
y = ['dialogue'] * dialogue_df.shape[0] + ['treatment'] * df.shape[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0) # test_size proportion 1:9
print('Train size = {}, test size = {}'.format(len(X_train), len(X_test)))

X_train_tfidf, X_test_tfidf = tfidf_features(X_train, X_test, RESOURCE_PATH['TFIDF_VECTORIZER'])

intent_recognizer = LogisticRegression(solver='lbfgs',penalty='l2',C=10,random_state=0,max_iter=400).fit(X_train_tfidf, y_train)

# Check test accuracy.
y_test_pred = intent_recognizer.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('Test accuracy = {}'.format(test_accuracy)) #acc = 0.9966555183946488

# Dump do classificador para depois usar no Bot
pickle.dump(intent_recognizer, open(RESOURCE_PATH['INTENT_RECOGNIZER'], 'wb'))
'''
'''--------------------------------- Intent recgonizer ---------------------------------'''

'''--------------------------------- Tag Classifier ---------------------------------'''
'''
X = df['treatment'].values
y = df['disease'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print('Train size = {}, test size = {}'.format(len(X_train), len(X_test)))

vectorizer = pickle.load(open(RESOURCE_PATH['TFIDF_VECTORIZER'], 'rb'))

X_train_tfidf, X_test_tfidf = vectorizer.transform(X_train), vectorizer.transform(X_test)

tag_classifier = OneVsRestClassifier(LogisticRegression(solver='lbfgs',penalty='l2',C=5,random_state=0,max_iter=400)).fit(X_train_tfidf, y_train)

# Check test accuracy.
y_test_pred = tag_classifier.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('Test accuracy = {}'.format(test_accuracy))

# Dump do classificador para depois usar no Bot
pickle.dump(tag_classifier, open(RESOURCE_PATH['TAG_CLASSIFIER'], 'wb'))
'''

'''--------------------------------- Tag Classifier ---------------------------------'''

'''--------------------------------- Rank com embeddings (temos de analisar) ---------------------------------'''
'''
counts_by_tag = df.groupby(by=['crop', 'disease'])['treatment'].count()
print(counts_by_tag)

os.makedirs(RESOURCE_PATH['THREAD_EMBEDDINGS_FOLDER'], exist_ok=True)

for crop, disease, count in counts_by_tag.items():
    tag_treatment = df[df['disease'] == disease]

    tag_post_ids = tag_treatment['treatment'].tolist()

    tag_vectors = np.zeros((count, embeddings_dim), dtype=np.float32)
    for i, title in enumerate(tag_treatment['treatment']):
        tag_vectors[i, :] = question_to_vec(title, starspace_embeddings, embeddings_dim)

    # Dump post ids and vectors to a file.
    filename = os.path.join(RESOURCE_PATH['THREAD_EMBEDDINGS_FOLDER'], os.path.normpath('%s.pkl' % disease))
    pickle.dump((tag_post_ids, tag_vectors), open(filename, 'wb'))
'''
'''--------------------------------- Rank com embeddings (temos de analisar) ---------------------------------'''

pd.set_option('display.max_colwidth', None)
df = pd.read_csv("new.csv", sep=";")

testOneDisease = "cropcommonrust " + df[df['disease'] == 'Common Rust']['treatment'].apply(text_prepare) + " EOF"

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


X_text = expanded_df['Leading Words']
y = expanded_df['Next Word']  # Replace 'labels' with the actual column name for your categories

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(X_text).toarray()

# Convert TF-IDF matrices to sequences
X_sequences = [np.where(row > 0)[0] for row in X_tfidf]

# Pad sequences to have a consistent length
max_sequence_length = max(len(seq) for seq in X_sequences)
X_padded = pad_sequences(X_sequences, maxlen=max_sequence_length)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y)

X_padded = np.reshape(X_padded, (X_padded.shape[0], X_padded.shape[1], 1))

# Define the RNN model
model = Sequential()
model.add(LSTM(128))
model.add(Dense(len(np.unique(y))))
model.add(Activation('softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Function to generate text
def on_epoch_end(epoch, _):
    if epoch == 58:
        print()
        print('----- Generating text after Epoch: %d' % epoch)
        #start_index = random.randint(0, len(testOneDisease) - maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('----- diversity:', diversity)

            generated = ''
            sentence = "cropcommonrust"
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            while True or len(generated) < 100:
                vectTest = vectorizer.transform([generated]).toarray()
                vectTest_sequences = [np.where(row > 0)[0] for row in vectTest]

                # Pad sequences to have a consistent length
                vectTest_padded = pad_sequences(vectTest_sequences, maxlen=max_sequence_length)
                print(vectTest_padded)

                nextChar = model.predict(vectTest_padded).argmax(axis=1)
                print(nextChar)
                nextWord = label_encoder.inverse_transform(nextChar)[0]
                generated += " " + nextWord
                print(generated)
                if nextChar == "EOF":
                    break

# Callback to generate text after each epoch
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)


# Train the model
model.fit(X_padded, y_train, batch_size=128, epochs=60, callbacks=[print_callback])

'''
# Extract text from the 'treatment' column
text = ' '.join(df['treatment'].apply(text_prepare).astype(str))

# Create character-level mappings
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
print(indices_char)


# Create training data
maxlen = 40
step = 3
sentences = []
next_chars = []
for x in testOneDisease:
    for i in range(0, len(x) - maxlen, step):
        sentences.append(x[i: i + maxlen])
        next_chars.append(x[i + maxlen])
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=bool)
    y = np.zeros((len(sentences), len(chars)), dtype=bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

#print(x)
#print(next_chars)


# Define the RNN model
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

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
    if epoch == 58:
        print()
        print('----- Generating text after Epoch: %d' % epoch)
        start_index = random.randint(0, len(testOneDisease) - maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('----- diversity:', diversity)

            generated = ''
            sentence = testOneDisease[start_index][:maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(400):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
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
'''
