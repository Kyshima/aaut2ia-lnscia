import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf
import pickle

def load_image(image_path):
    image = Image.open(image_path)
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

def predict_crop(image_path):
    model_path = "C:/Users/Diana/Documents/aaut2ia-lnscia/MachineLearning/NeuralNetwork/Crop/Final_Models/Models_Exported/crop3.h5"
    label_encoder_path = "C:/Users/Diana/Documents/aaut2ia-lnscia/MachineLearning/Predict/label_encoder_crop.pkl"

    model = tf.keras.models.load_model(model_path)
    label_encoder = pickle.load(open(label_encoder_path, "rb"))

    image_array = load_image(image_path)

    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)

    crop = label_encoder.inverse_transform([predicted_class])[0]
    return crop

image_path = "C:/Users/Diana/Desktop/00d8f10f-5038-4e0f-bb58-0b885ddc0cc5___RS_Early.B 8722.jpg"
predicted_crop = predict_crop(image_path)
print(predicted_crop)
