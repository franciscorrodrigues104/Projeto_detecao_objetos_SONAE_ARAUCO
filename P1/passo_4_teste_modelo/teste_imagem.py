from ultralytics import YOLO
import os

caminho = r"C:\Users\frrodrigues\Desktop\Projeto_detecao_trator\passo_4_teste_modelo\best.pt"
model = YOLO(caminho)

results = model(r"C:\Users\frrodrigues\Desktop\proj2\frames\trator (85).jpg", verbose=False)

counter = 0
last_detection = 0
falhaDetetada = False


for box in results[0].boxes:
        cls = int(box.cls[0])
        if cls == 0:
            falhaDetetada = True
            counter += 1
            print("Falhas identificadas na imagem: ", counter)
                        

pasta_projeto = r"C:\Users\frrodrigues\Desktop\Projeto_detecao_trator\passo_4_teste_modelo"
caminho_saida = os.path.join(pasta_projeto, "img_detecao_5.jpg")
results[0].save(filename=caminho_saida)
print("Falhas identificadas na imagem: ", counter)

