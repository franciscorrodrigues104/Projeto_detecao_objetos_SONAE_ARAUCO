from ultralytics import YOLO
import os

caminho = r"C:\Users\frrodrigues.INDPT\Desktop\Projeto_detecao_trator\P1_Projeto_detecao_trator_recolha_madeira\passo_3_detecao_front_end\app_detecao\best.pt"
model = YOLO(caminho)

results = model(r"C:\Users\frrodrigues\OneDrive - Sonae Arauco\Pictures\Screenshots\Captura de ecrã 2026-08-21 154929.png", verbose=False, conf=0.1)

counter = 0
last_detection = 0
falhaDetetada = False


for box in results[0].boxes:
        cls = int(box.cls[0])
        if cls == 0:
            falhaDetetada = True
            counter += 1
            print("Tratores identificados na imagem: ", counter)
                        

pasta_projeto = r"C:\Users\frrodrigues\Desktop\Projeto_detecao_trator\P1_Projeto_detecao_trator_recolha_madeira\passo_2_teste_modelo"
caminho_saida = os.path.join(pasta_projeto, "img_detecao_5.jpg")
results[0].save(filename=caminho_saida)
print("Falhas identificadas na imagem: ", counter)

