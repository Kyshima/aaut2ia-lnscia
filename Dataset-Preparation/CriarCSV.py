import os
import cv2
import pandas as pd

def obter_informacoes_pixels(imagem):
    altura, largura, _ = imagem.shape
    pixels = imagem.reshape(-1, 3)
    return pixels, largura, altura

def criar_csv_com_informacoes(diretorio_dos_dados):
    numbro = 0
    dados = []

    for pasta, subpastas, arquivos in os.walk(diretorio_dos_dados):
        for nome_do_arquivo in arquivos:
            print(numbro, nome_do_arquivo)
            numbro = numbro + 1
            if nome_do_arquivo.lower().endswith('.jpg'):
                caminho_da_imagem = os.path.join(pasta, nome_do_arquivo)
                imagem = cv2.imread(caminho_da_imagem)
                pixels, largura, altura = obter_informacoes_pixels(imagem)
                informacao_pixels = [' '.join(map(str, pixel)) for pixel in pixels]
                dados.append([pasta, nome_do_arquivo, largura, altura, informacao_pixels])

    df = pd.DataFrame(dados, columns=['Pasta', 'Nome da Imagem', 'Largura', 'Altura', 'Informacao de Pixels'])
    df.to_csv('imagens.csv', index=False)

if __name__ == "__main__":
    diretorio_dos_dados = r'DataSet\Teste'
    criar_csv_com_informacoes(diretorio_dos_dados)
