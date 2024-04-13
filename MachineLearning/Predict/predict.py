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

def predict_crop(image_array):
    model_path = "C:/Users/Diana/Documents/aaut2ia-lnscia/MachineLearning/NeuralNetwork/Crop/Final_Models/Models_Exported/crop2.h5"
    label_encoder_path = "C:/Users/Diana/Documents/aaut2ia-lnscia/MachineLearning/Predict/label_encoder_crop.pkl"

    model = tf.keras.models.load_model(model_path)
    label_encoder = pickle.load(open(label_encoder_path, "rb"))

    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)

    crop = label_encoder.inverse_transform([predicted_class])[0]
    return crop

def predict_illness(image_array):
    model_path = "C:/Users/Diana/Documents/aaut2ia-lnscia/MachineLearning/NeuralNetwork/Illness/Final_Models/Models_Exported/illness3.h5"
    label_encoder_path = "C:/Users/Diana/Documents/aaut2ia-lnscia/MachineLearning/Predict/label_encoder_illness.pkl"

    model = tf.keras.models.load_model(model_path)
    label_encoder = pickle.load(open(label_encoder_path, "rb"))

    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)

    illness = label_encoder.inverse_transform([predicted_class])[0]
    return illness

def predict_crop_illness(image_array):
    model_path = "C:/Users/Diana/Documents/aaut2ia-lnscia/MachineLearning/NeuralNetwork/Crop_Illness/Final_Models/Models_Exported/crop_illness2.h5"
    label_encoder_path = "C:/Users/Diana/Documents/aaut2ia-lnscia/MachineLearning/Predict/label_encoder_crop_illness.pkl"

    model = tf.keras.models.load_model(model_path)
    label_encoder = pickle.load(open(label_encoder_path, "rb"))

    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)

    crop_illness = label_encoder.inverse_transform([predicted_class])[0]
    return crop_illness


#"C:\Users\Diana\Desktop\DataSet\CropDisease\Crop___DIsease\Corn___Gray_Leaf_Spot\ff8671f5-09be-49d7-8093-2707c3a32489___RS_GLSp 4620_new30degFlipLR.JPG"

image_path = "C:/Users/Diana/Desktop/DataSet/CropDisease/Crop___DIsease/Wheat___Brown_Rust/Brown_rust711.JPG"
image_array = load_image(image_path)
predicted_crop = predict_crop(image_array)
predicted_illness = predict_illness(image_array)
predicted_crop_illness = predict_crop_illness(image_array)

print(predicted_crop)
print(predicted_illness)
print(predicted_crop_illness)