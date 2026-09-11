import os

# Caminho para a pasta principal 'mesa1_tras'
pasta_principal = r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_1_extracao_imagens_frames\pt4_melhoria"

caminhos_mkv = []

# Percorre todas as pastas, subpastas e ficheiros
for raiz, diretorios, ficheiros in os.walk(pasta_principal):
    for ficheiro in ficheiros:
        # Filtra apenas os ficheiros que terminam em .mkv
        if ficheiro.lower().endswith(".mkv"):
            # Obtém o caminho absoluto completo
            caminho_absoluto = os.path.join(raiz, ficheiro)
            caminhos_mkv.append(caminho_absoluto)

# Mostra os resultados na consola
print(f"Foram encontrados {len(caminhos_mkv)} vídeos .mkv:\n")
for i, caminho in enumerate(caminhos_mkv, 1):
    # Ajustado para imprimir exatamente r"caminho"
    print(f',r"{caminho}"')