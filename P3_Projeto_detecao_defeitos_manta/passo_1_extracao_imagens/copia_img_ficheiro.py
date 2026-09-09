import os
import shutil

# 1. Configurações de pastas
pasta_origem = r"C:\Users\frrodrigues\Desktop\projeto_defeitos_manta\passo_2_teste_modelo\resultados_testes"
pasta_destino = r"C:\Users\frrodrigues\Desktop\projeto_defeitos_manta\passo_1_extracao_imagens\dataset_defeitos_v1_bg"

# 2. Lista de números a copiar
lista_numeros = [
1, 2, 3, 4, 5, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 35, 36, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 106, 107, 118, 119, 260, 262, 263, 264, 265, 266, 267, 281, 282, 283, 288, 289, 292, 315, 320, 321, 339, 347, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385
]

# Garantir que a pasta destino existe
os.makedirs(pasta_destino, exist_ok=True)

contador = 0

# 3. Processo de cópia
for numero in lista_numeros:
    # O nome esperado é "falhas (x).jpg" ou "falhas (x).png"
    # Ajusta a extensão abaixo se necessário
    nome_ficheiro = f"bg_v1 ({numero}).png" 
    caminho_origem = os.path.join(pasta_origem, nome_ficheiro)
    
    if os.path.exists(caminho_origem):
        shutil.copy2(caminho_origem, pasta_destino)
        print(f"Copiado: {nome_ficheiro}")
        contador += 1
    else:
        print(f"Aviso: Ficheiro não encontrado: {nome_ficheiro}")

print(f"\nConcluído! Foram copiadas {contador} imagens.")