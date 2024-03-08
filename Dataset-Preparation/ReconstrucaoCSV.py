import os
import numpy as np
import cv2
import pandas as pd

def reconstruir_imagem(caminho_csv, indice):
    df = pd.read_csv(caminho_csv)

    linha = df.iloc[indice]
    nome_da_imagem = 'Teste.jpg'
    largura = linha['Largura']
    altura = linha['Altura']

    informacao_pixels = [list(map(int, pixel.split())) for pixel in linha['Informacao de Pixels'].strip("[]").replace("'", "").split(',')]
    imagem_reconstruida = np.array(informacao_pixels, dtype=np.uint8).reshape(altura, largura, 3)

    caminho_saida = os.path.join(f"reconstruida_{nome_da_imagem}")
    print(caminho_saida)
    cv2.imwrite(caminho_saida, imagem_reconstruida)

if __name__ == "__main__":
    caminho_csv = 'Dataset.csv'
    indice_a_reconstruir = 1  # índice da linha a reconstruir
    reconstruir_imagem(caminho_csv, indice_a_reconstruir)
