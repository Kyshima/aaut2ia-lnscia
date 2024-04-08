import numpy as np
import pandas as pd
import pickle
import re
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


sample_size = 3495

dialogue_df = pd.read_csv('data/dialogues.tsv', sep='\t').sample(sample_size, random_state=0)
crop_dialogue_df = pd.read_csv('data/crop_dialogue.csv', sep=';')


dialogue_df['text'] = dialogue_df['text'].apply(text_prepare)
crop_dialogue_df['visual_description'] = crop_dialogue_df['visual_description'].apply(text_prepare)


X = np.concatenate([dialogue_df['text'].values, crop_dialogue_df['visual_description'].values])
y = ['dialogue'] * dialogue_df.shape[0] + ['visual_description'] * crop_dialogue_df.shape[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0) # test_size proportion 1:9
print('Train size = {}, test size = {}'.format(len(X_train), len(X_test)))

X_train_tfidf, X_test_tfidf = tfidf_features(X_train, X_test, RESOURCE_PATH['TFIDF_VECTORIZER'])

intent_recognizer = LogisticRegression(solver='lbfgs',penalty='l2',C=10,random_state=0,max_iter=400).fit(X_train_tfidf, y_train)

# Check test accuracy.
y_test_pred = intent_recognizer.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('Test accuracy = {}'.format(test_accuracy))

pickle.dump(intent_recognizer, open(RESOURCE_PATH['INTENT_RECOGNIZER'], 'wb'))