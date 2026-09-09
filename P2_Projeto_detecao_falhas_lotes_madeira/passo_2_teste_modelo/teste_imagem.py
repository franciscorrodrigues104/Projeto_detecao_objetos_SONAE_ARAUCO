from ultralytics import YOLO
import os


caminho = r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_3_teste_modelo\pt2\v12_a_partir_do_v10.pt"
model = YOLO(caminho)

results = model(r"C:\Users\frrodrigues.INDPT\Downloads\cam1_original_2026-09-07_13-49-23.jpg", verbose=False, conf=0.1)

counter = 0
last_detection = 0
falhaDetetada = False


for box in results[0].boxes:
        cls = int(box.cls[0])
        if cls == 2:
            conf_score = float(box.conf[0])
            falhaDetetada = True
            counter += 1
            print("Falhas identificadas na imagem: ", counter, " - score: ", conf_score)
                        

pasta_projeto = r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_3_teste_modelo\pt2\files"
caminho_saida = os.path.join(pasta_projeto, "img_teste_v12.jpg")
results[0].save(filename=caminho_saida)

