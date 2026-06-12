#se utilizar uma ferramenta como o robflow não precisa deste passo
import os, random, shutil

origem = r"C:\Users\frrodrigues\Desktop\Projeto_detecao_trator\passo_1_extracao_imagens_frames\dataset"
destino = r
proporcao_treino = 0.8

for pasta in ['train', 'val']:
    os.makedirs(os.path.join(destino, pasta, 'images'), exist_ok=True)
    os.makedirs(os.path.join(destino, pasta, 'labels'), exist_ok=True)

fotos = [f for f in os.listdir(origem) if f.endswith('.jpg')]
random.shuffle(fotos)

split = int(len(fotos) * proporcao_treino)

for i, nome_foto in enumerate(fotos):
    subpasta = 'train' if i < split else 'val'
    nome_txt = nome_foto.replace('.jpg', '.txt')

    shutil.copy(os.path.join(origem, nome_foto), os.path.join(destino, subpasta, 'images', nome_foto))
    
    # Mover TXT (se existir)
    caminho_txt = os.path.join(origem, nome_txt)
    if os.path.exists(caminho_txt):
        shutil.copy(caminho_txt, os.path.join(destino, subpasta, 'labels', nome_txt))

print("Pronto! Fotos e Labels separados.")