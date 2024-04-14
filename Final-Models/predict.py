from utils import *
import tensorflow as tf
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from PIL import Image, ImageEnhance

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



def load_image(image):
    image = Image.fromarray(np.array(image))
    image = image.convert('RGB')
    image = image.resize((128, 128))
    image_array = np.array(image)
    color_enhancer = ImageEnhance.Color(image)
    image = color_enhancer.enhance(10)

    r, g, b = image.split()

    factor = 0.7
    g_less_strong = g.point(lambda x: int(x * factor))

    image = Image.merge("RGB", (r, g_less_strong, b))

    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_crop(image, type):
    if(type == 1):
        model_path = "models/crop2.h5"
        label_encoder_path = "models/label_encoder_crop.pkl"
    if(type == 2):
        model_path = "models/illness3.h5"
        label_encoder_path = "models/label_encoder_illness.pkl"
    if (type == 3):
        model_path = "models/crop_illness2.h5"
        label_encoder_path = "models/label_encoder_crop_illness.pkl"

    model = tf.keras.models.load_model(model_path)
    label_encoder = pickle.load(open(label_encoder_path, "rb"))

    image_array = load_image(image)

    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)

    crop = label_encoder.inverse_transform([predicted_class])[0]
    return crop


