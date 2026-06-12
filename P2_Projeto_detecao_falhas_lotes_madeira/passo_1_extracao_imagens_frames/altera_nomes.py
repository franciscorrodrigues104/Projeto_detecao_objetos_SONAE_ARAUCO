import os
import shutil

def organizar_e_renomear(pasta_origem, pasta_destino, prefixo):
    # Cria a pasta de destino se não existir
    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)
    
    # Lista todos os arquivos da pasta de origem
    arquivos = os.listdir(pasta_origem)
    
    # Filtra apenas imagens
    extensoes_validas = ('.jpg', '.jpeg', '.png', '.bmp')
    imagens = [f for f in arquivos if f.lower().endswith(extensoes_validas)]
    imagens.sort() # Garante a ordem alfabética original
    
    print(f"Encontradas {len(imagens)} imagens na pasta '{pasta_origem}'...")
    
    for i, nome_antigo in enumerate(imagens, start=1):
        extensao = os.path.splitext(nome_antigo)[1]
        nome_novo = f"{prefixo}_{i}{extensao}"
        
        # Caminhos completos
        caminho_antigo = os.path.join(pasta_origem, nome_antigo)
        caminho_novo = os.path.join(pasta_destino, nome_novo)
        
        # Copia e renomeia (usamos copy para não apagar o original)
        shutil.copy(caminho_antigo, caminho_novo)
        print(f"Copiado e Renomeado: {nome_novo}")

# --- CONFIGURAÇÃO ---
# Pode alterar estes caminhos para os seus caminhos reais
origem = r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_1_extracao_imagens_frames\novas_imagens\v6_dataset_falhas\falhas" 
destino = r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_1_extracao_imagens_frames\novas_imagens\v6_dataset_falhas"

organizar_e_renomear(origem, destino, "falhas")