import os

import PIL
import numpy
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import cv2


def obter_informacoes_pixels(imagem):
    altura, largura, _ = imagem.shape
    pixels = imagem.reshape(-1, 3)
    return pixels, largura, altura

def processar_imagem(caminho_da_imagem):
    imagem = Image.open(caminho_da_imagem)
    color_enhancer = ImageEnhance.Color(imagem)
    image = color_enhancer.enhance(10)  # Increase color intensity

    image_rgb = image.convert("RGB")

    # Separate the channels
    r, g, b = image_rgb.split()

    # Reduce the strength of the green channel
    factor = 0.7  # Adjust this factor as needed (0.5 reduces intensity by half)
    g_less_strong = g.point(lambda x: int(x * factor))

    # Recombine the channels into an image
    imagem = Image.merge("RGB", (r, g_less_strong, b))

    open_cv_image = numpy.array(imagem.convert('RGB'))
    # Convert RGB to BGR
    imagem = open_cv_image[:, :, ::-1].copy()
    imagem = cv2.resize(imagem, (128, 128))
    pixels, largura, altura = obter_informacoes_pixels(imagem)
    informacao_pixels = pickle.dumps(pixels)
    return informacao_pixels

count = 0
def criar_dataframe_arquivo(pasta, nome_do_arquivo):
    global count
    crop_illness = pasta.split("\\")[-1]
    if crop_illness != "Invalid" and "(1)" not in nome_do_arquivo and nome_do_arquivo.lower().endswith('.jpg'):
        caminho_da_imagem = os.path.join(pasta, nome_do_arquivo)
        informacao_pixels = processar_imagem(caminho_da_imagem)
        count += 1
        if count % 100 == 0:
            print(count)
        return pd.DataFrame({
            'crop': crop_illness.split("___")[0],
            'illness': crop_illness.split("___")[-1],
            'crop_illness': crop_illness,
            'Informacao de Pixels': informacao_pixels,
        }, index=[0])
    return None


def criar_csv_com_informacoes(diretorio_dos_dados):
    header = True
    dadosMaster = []
    for pasta, subpastas, arquivos in os.walk(diretorio_dos_dados):
        print(pasta)
        dados = [criar_dataframe_arquivo(pasta, nome_do_arquivo) for nome_do_arquivo in arquivos]
        dados = [df for df in dados if df is not None][:500]
        print(len(dados))
        dadosMaster = dadosMaster + dados
        if len(dadosMaster) >= 500:
            break

    dadosMaster = pd.concat(dadosMaster, axis=0, ignore_index=True)
    dadosMaster.fillna(0, inplace=True)
    label_encoder = LabelEncoder()
    dadosMaster['crop'] = label_encoder.fit_transform(dadosMaster['crop'])
    dadosMaster['illness'] = label_encoder.fit_transform(dadosMaster['illness'])
    dadosMaster['crop_illness'] = label_encoder.fit_transform(dadosMaster['crop_illness'])
    dadosMaster.to_csv("DatasetBinary128.csv", mode='a', header=header, index=False)
    header = False


if __name__ == "__main__":
    diretorio_dos_dados = r'C:/Users/Patricia/Documents/GitHub/aaut2ia-lnscia/Dataset'
    criar_csv_com_informacoes(diretorio_dos_dados)