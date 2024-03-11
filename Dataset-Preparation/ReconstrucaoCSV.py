import os
import numpy as np
import cv2
import pandas as pd
import pickle

def reconstruir_imagem(caminho_csv, indice, largura, altura):
    df = pd.read_csv(caminho_csv)

    linha = df.iloc[indice]
    nome_da_imagem = 'Teste.jpg'

    informacao_pixels = pickle.loads(eval(linha['Informacao de Pixels']))
    imagem_reconstruida = np.array(informacao_pixels, dtype=np.uint8).reshape(largura, altura, 3)

    caminho_saida = os.path.join(f"{nome_da_imagem}")
    print(caminho_saida)
    cv2.imwrite(caminho_saida, imagem_reconstruida)

if __name__ == "__main__":
    caminho_csv = 'DatasetBinary128.csv'
    indice_a_reconstruir = 178  # índice da linha a reconstruir
    largura, altura = 128, 128
    reconstruir_imagem(caminho_csv, indice_a_reconstruir, largura, altura)
