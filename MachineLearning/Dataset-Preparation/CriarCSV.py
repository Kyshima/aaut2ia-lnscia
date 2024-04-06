import os
import numpy as np
import pandas as pd
import pickle
import cv2
from PIL import Image, ImageEnhance
from sklearn.preprocessing import LabelEncoder

def get_pixel_info(image):
    height, width, _ = image.shape
    pixels = image.reshape(-1, 3)
    return pixels, width, height

def process_image(image_path):
    image = Image.open(image_path)
    color_enhancer = ImageEnhance.Color(image)
    image = color_enhancer.enhance(10)  # Increase color intensity

    image_rgb = image.convert("RGB")

    # Separate the channels
    r, g, b = image_rgb.split()

    # Reduce the strength of the green channel
    factor = 0.7  # Adjust this factor as needed (0.5 reduces intensity by half)
    g_less_strong = g.point(lambda x: int(x * factor))

    # Recombine the channels into an image
    image = Image.merge("RGB", (r, g_less_strong, b))

    open_cv_image = np.array(image.convert('RGB'))
    # Convert RGB to BGR
    image = open_cv_image[:, :, ::-1].copy()
    image = cv2.resize(image, (128, 128))
    pixels, width, height = get_pixel_info(image)
    pixel_info = pickle.dumps(pixels)
    return pixel_info

count = 0
def create_dataframe_file(folder, file_name):
    global count
    crop_illness = folder.split("\\")[-1]
    if crop_illness != "Invalid" and "(1)" not in file_name and file_name.lower().endswith('.jpg'):
        image_path = os.path.join(folder, file_name)
        pixel_info = process_image(image_path)
        count += 1
        if count % 100 == 0:
            print(count)
        return pd.DataFrame({
            'crop': crop_illness.split("___")[0],
            'illness': crop_illness.split("___")[-1],
            'crop_illness': crop_illness,
            'Informacao de Pixels': pixel_info,
        }, index=[0])
    return None

def create_csv_with_info(data_directory):
    header = True
    master_data = []
    for folder, subfolders, files in os.walk(data_directory):
        print(folder)
        data = [create_dataframe_file(folder, file_name) for file_name in files]
        data = [df for df in data if df is not None][:500]
        print(len(data))
        master_data = master_data + data

    master_data = pd.concat(master_data, axis=0, ignore_index=True)
    master_data.fillna(0, inplace=True)
    label_encoder = LabelEncoder()
    master_data['crop'] = label_encoder.fit_transform(master_data['crop'])
    master_data['illness'] = label_encoder.fit_transform(master_data['illness'])
    master_data['crop_illness'] = label_encoder.fit_transform(master_data['crop_illness'])
    master_data.to_csv("DatasetBinary128.csv", mode='a', header=header, index=False)
    header = False

if __name__ == "__main__":
    data_directory = r'C:/Users/Diana/Desktop/Dataset'
    create_csv_with_info(data_directory)
