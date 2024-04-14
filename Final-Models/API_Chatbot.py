import json
import numpy as np
import pandas as pd
import random
from flask import Flask, request, jsonify
from transformers import pipeline
from transformers import BertForSequenceClassification, BertTokenizerFast
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

#Load data
def load_json_file(filename):
    with open(filename) as f:
        file = json.load(f)
    return file

filename = 'datasets/intents.json'
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
crop_dialogue_df = pd.read_csv('datasets/crop_dialogue.csv', sep=';')
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


model_path = "chatbot"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer= BertTokenizerFast.from_pretrained(model_path)
chatbot= pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data = request.get_json()
    text = data.get('text').strip().lower()

    give_treatment = data.get('give_treatment').strip().lower()

    score = chatbot(text)[0]['score']

    give_description = 'no'

    if give_treatment == "yes":
        data = {'text': text}

        response_treatment = requests.post("http://localhost:5000/language_predict", json=data)

        if response_treatment.status_code == 200:
            result = response_treatment.json()

            disease = ''
            if result['disease'] == 'cornleafblight':
                disease = 'Corn - Leaf Blight'
            elif result['disease'] == 'corncommonrust':
                disease = 'Corn - Common Rust'
            elif result['disease'] == 'corngrayleafspot':
                disease = 'Corn - Gray Leaf Spot'
            elif result['disease'] == 'cornleafblight':
                disease = 'Corn - Leaf Blight'
            elif result['disease'] == 'potatoearlyblight':
                disease = 'Potato - Early Blight'
            elif result['disease'] == 'potatolateblight':
                disease = 'Potato - Late Blight'
            elif result['disease'] == 'ricebrownspot':
                disease = 'Rice - Brown Spot'
            elif result['disease'] == 'ricehispa':
                disease = 'Rice - Hispa'
            elif result['disease'] == 'riceleafblast':
                disease = 'Rice - Leaf Blast'
            elif result['disease'] == 'wheatbrownrust':
                disease = 'Wheat - Brown Rust'
            elif result['disease'] == 'wheatyellowrust':
                disease = 'Wheat - Yellow Rust'

            disease_treatment = disease + ': ' + result['treatment']

            response = disease_treatment
        else:
            print('Failed to retrieve prediction')

        give_description = 'no'
    else:
        if score < 0.8:
            response = "Sorry I was not trained with an adequate response for that. For more info, please contact my makers: https://grupo6meia.wixsite.com/grupo-6"
        else:
            label = label2id[chatbot(text)[0]['label']]
            if label == 0:
                response = "Could you give me visual description of your crop leaf?"
                give_description = 'yes'
            else:
                give_description = 'no'
                response = random.choice(intents['intents'][label-1]['responses'])

    return jsonify({'response': response, 'give_description': give_description})

if __name__ == '__main__':
    app.run(host='localhost', port=5001)