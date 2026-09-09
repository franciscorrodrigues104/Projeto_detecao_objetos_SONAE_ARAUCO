import cv2

# Usa sempre o 'r' antes das aspas no caminho do Windows
caminho_imagem = r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_4_teste_modelo\pt2\files_modelo\QFWRWQG.jpg"

# 1. Carregar a imagem a cores
imagem_colorida = cv2.imread(caminho_imagem)

# 2. Verificar se a imagem foi realmente lida
if imagem_colorida is None:
    print(f"Erro: Não foi possível carregar a imagem no caminho: {caminho_imagem}")
    print("Verifica se o caminho está correto e se o nome do ficheiro não tem caracteres problemáticos.")
else:
    # 3. Converter para escala de cinzentos
    imagem_pb = cv2.cvtColor(imagem_colorida, cv2.COLOR_BGR2GRAY)

    # 4. Guardar a nova imagem
    caminho_saida = r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_4_teste_modelo\pt2\files_modelo\imagem_pb.jpg"
    cv2.imwrite(caminho_saida, imagem_pb)
    print(f"Imagem convertida e guardada com sucesso em: {caminho_saida}")