import warnings
warnings.filterwarnings('ignore')
import json
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS
import missingno as msno

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import keras
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,Dropout
from keras.callbacks import ReduceLROnPlateau

import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import pipeline
from transformers import DistilBertTokenizerFast
from transformers import BertForSequenceClassification, BertTokenizerFast
from transformers import TFDistilBertForSequenceClassification, TFTrainingArguments
from transformers import BertTokenizer, TFBertForSequenceClassification, BertConfig
from transformers import TrainingArguments, Trainer


#Load data
def load_json_file(filename):
    with open(filename) as f:
        file = json.load(f)
    return file

filename = 'intents.json'
intents = load_json_file(filename)
def create_df():
    df = pd.DataFrame({
        'Pattern': [],
        'Tag': []
    })

    return df

df = create_df()
def extract_json_info(json_file, df):
    for intent in json_file['intents']:

        for pattern in intent['patterns']:
            sentence_tag = [pattern, intent['tag']]
            df.loc[len(df.index)] = sentence_tag

    return df

def extract_df_info(df_info, df):
    for pattern in df_info['visual_description']:
        sentence_tag = [pattern, 'crop']
        df.loc[len(df.index)] = sentence_tag

    return df


#Tags de preocupação com crops
crop_dialogue_df = pd.read_csv('../chatbot/data/crop_dialogue.csv', sep=';')
df = extract_df_info(crop_dialogue_df, df)

#Tags de chitchat
df = extract_json_info(intents, df)


#Labeling
df2 = df.copy()
labels = df2['Tag'].unique().tolist()
labels = [s.strip() for s in labels]

num_labels = len(labels)
id2label = {id:label for id, label in enumerate(labels)}
label2id = {label:id for id, label in enumerate(labels)}

df2['labels'] = df2['Tag'].map(lambda x: label2id[x.strip()])

#Test Split
X = list(df2['Pattern'])
y = list(df2['labels'])

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 123)

'''
#BERT Model
model_name = "bert-base-uncased"
max_len = 256
tokenizer = BertTokenizer.from_pretrained(model_name,
                                          max_length=max_len)
model = BertForSequenceClassification.from_pretrained(model_name,
                                                      num_labels=num_labels,
                                                      id2label=id2label,
                                                      label2id = label2id)


#Tokenize
train_encoding = tokenizer(X_train, truncation=True, padding=True)
test_encoding = tokenizer(X_test, truncation=True, padding=True)
full_data = tokenizer(X, truncation=True, padding=True)


#Data Loader
class DataLoader(Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataloader = DataLoader(train_encoding, y_train)
test_dataloader = DataLoader(test_encoding, y_test)
fullDataLoader = DataLoader(full_data, y_test)


#Metrics Evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)

    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }

training_args = TrainingArguments(
    output_dir='./output2',
    do_train=True,
    do_eval=True,
    num_train_epochs=50,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.05,
    logging_strategy='steps',
    logging_dir='./multi-class-logs2',
    logging_steps=25,
    evaluation_strategy="steps",
    eval_steps=25,
    save_strategy="steps",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataloader,
    eval_dataset=test_dataloader,
    compute_metrics= compute_metrics
)

trainer.train()


model_path = "chatbot2"
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)

'''

model_path = "chatbot2"

model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer= BertTokenizerFast.from_pretrained(model_path)
chatbot= pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

print(chatbot("fuck"))


def chat(chatbot):
    print(
        "Chatbot: Hi! I am your virtual assistance,Feel free to ask, and I'll do my best to provide you with answers and assistance..")
    print("Type 'quit' to exit the chat\n\n")

    text = input("User: ").strip().lower()

    while (text != 'quit'):

        score = chatbot(text)[0]['score']

        if score < 0.8:
            print("Chatbot: Sorry I can't answer that\n\n")
            text = input("User: ").strip().lower()
            continue

        label = label2id[chatbot(text)[0]['label']]

        if label == 0:
            response = "Tratamento maravilha"
        else:
            response = random.choice(intents['intents'][label-1]['responses'])

        print(f"Chatbot: {response}\n\n")

        text = input("User: ").strip().lower()


chat(chatbot)