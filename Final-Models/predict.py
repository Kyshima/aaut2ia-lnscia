from utils import *
import tensorflow as tf
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

def predict_formality(text):
    with open("models/formality-model-vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("models/formality-model-sequence_length.pkl", "rb") as f:
        max_sequence_length = pickle.load(f)

    with open("models/formality-model-encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    X_tfidf = vectorizer.transform([text_prepare(text, True)]).toarray()

    # Convert TF-IDF matrices to sequences
    X_sequences = [np.where(row > 0)[0] for row in X_tfidf]


    X_padded = pad_sequences(X_sequences, maxlen=max_sequence_length)

    model = load_model("models/formality-model.h5")
    formality = label_encoder.inverse_transform(model.predict(X_padded).argmax(axis=1))

    return formality[0]


def predict_disease(text, formality):

    with open("models/disease-classification-vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("models/disease-classification-sequence_length.pkl", "rb") as f:
        max_sequence_length = pickle.load(f)

    with open("models/disease-classification-formality-encoder.pkl", "rb") as f:
        formality_encoder = pickle.load(f)

    with open("models/disease-classification-label-encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    form = formality_encoder.transform([formality])

    X_tfidf = vectorizer.transform([text_prepare(text, False)]).toarray()

    # Convert TF-IDF matrices to sequences
    X_sequences = [np.where(row > 0)[0] for row in X_tfidf]

    X_padded = pad_sequences(X_sequences, maxlen=max_sequence_length)
    X = np.hstack((X_padded, [form]))

    model = load_model("models/disease-classification-model.h5")
    disease = label_encoder.inverse_transform(model.predict(X_padded).argmax(axis=1))

    return disease[0]

def generate_text(disease):

    with open("models/text-generator-tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("models/text-generator-label-encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)


    model = load_model("models/text-generator-model.keras")

    generated = ''
    sentence = disease
    generated += sentence

    while True or len(generated) < 100:
        sequence = tokenizer.texts_to_sequences([generated])
        padded_sequence = pad_sequences(sequence, maxlen=100)
        predicted_index = np.argmax(model.predict(padded_sequence), axis=-1)[0]
        predicted_disease = label_encoder.inverse_transform([predicted_index])[0]

        generated += " " + predicted_disease
        if predicted_disease == "EOF":
            words = generated.split()

            if len(words) > 2:
                result = ' '.join(words[1:-1])
                return result
            else:
                return ""

def predict_language(text):
    formality = predict_formality(text)
    disease = predict_disease(text, formality)
    treatment = generate_text(disease)
    return disease, treatment

if __name__ == "__main__":
    disease, treatment = predict_language("The wheat leaves display small, rusty-brown spots scattered across their surface.")
    print(disease)
    print(treatment)


