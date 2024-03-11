import os
import cv2
import pandas as pd
import pickle

def obter_informacoes_pixels(imagem):
    altura, largura, _ = imagem.shape
    pixels = imagem.reshape(-1, 3)
    return pixels, largura, altura


def processar_imagem(caminho_da_imagem):
    imagem = cv2.imread(caminho_da_imagem)
    #imagem = cv2.resize(imagem, (256, 256))
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
            'crop': crop_illness.split("___")[1],
            'illness': crop_illness.split("___")[-1],
            'crop_illness': crop_illness,
            'Informacao de Pixels': [informacao_pixels],
        }, index=[0])
    return None


def criar_csv_com_informacoes(diretorio_dos_dados):
    header = True
    for pasta, subpastas, arquivos in os.walk(diretorio_dos_dados):
        print(pasta)
        dados = [criar_dataframe_arquivo(pasta, nome_do_arquivo) for nome_do_arquivo in arquivos]
        dados = [df for df in dados if df is not None]

        if dados:
            dados = pd.concat(dados, axis=0, ignore_index=True)
            dados.fillna(0, inplace=True)
            crop_dummies = pd.get_dummies(dados['crop'], prefix='crop', dtype=int)
            illness_dummies = pd.get_dummies(dados['illness'], prefix='illness', dtype=int)
            crop_illness_dummies = pd.get_dummies(dados['crop_illness'], prefix='crop_illness', dtype=int)
            dados = pd.concat([crop_dummies, illness_dummies, crop_illness_dummies, dados], axis=1)
            dados = dados.drop(columns=['crop', 'illness', 'crop_illness'])
            dados.to_csv("DatasetBinary128.csv", mode='a', header=header, index=False)
            header = False


if __name__ == "__main__":
    diretorio_dos_dados = r'DataSet'
    criar_csv_com_informacoes(diretorio_dos_dados)