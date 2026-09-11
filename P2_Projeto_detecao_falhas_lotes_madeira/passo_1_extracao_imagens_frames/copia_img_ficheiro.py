import os
import shutil

# 1. Configurações de pastas
pasta_origem = r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_1_extracao_imagens_frames\runs\detect\predict17\crops\lotes"
pasta_destino = r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_1_extracao_imagens_frames\pt4_melhoria\0"

# 2. Lista de números a copiar
lista_numeros = [
    1, 45, 51, 59, 65, 71, 78, 87, 94, 101, 110, 116, 121, 129, 143, 152, 158, 168, 175, 190, 197, 203, 209, 222, 227, 230, 237, 247, 254, 276, 301, 303
]

# Garantir que a pasta destino existe
os.makedirs(pasta_destino, exist_ok=True)

contador = 0

# 3. Processo de cópia
for numero in lista_numeros:
    # O nome esperado é "falhas (x).jpg" ou "falhas (x).png"
    # Ajusta a extensão abaixo se necessário
    nome_ficheiro = f"novos_v4 ({numero}).jpg" 
    caminho_origem = os.path.join(pasta_origem, nome_ficheiro)
    
    if os.path.exists(caminho_origem):
        shutil.copy2(caminho_origem, pasta_destino)
        print(f"Copiado: {nome_ficheiro}")
        contador += 1
    else:
        print(f"Aviso: Ficheiro não encontrado: {nome_ficheiro}")

print(f"\nConcluído! Foram copiadas {contador} imagens.")