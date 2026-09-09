from ultralytics import YOLO
import os


caminho = r"C:\Users\frrodrigues.INDPT\Desktop\projeto_defeitos_manta\passo_2_teste_modelo\modelo_detecao_defeitos_manta_v2.pt"
model = YOLO(caminho)

results = model(r"C:\Users\frrodrigues.INDPT\Desktop\projeto_defeitos_manta\passo_2_teste_modelo\manta_roi_3.png", verbose=False, conf=0.1)

counter = 0
last_detection = 0
falhaDetetada = False


for box in results[0].boxes:
        cls = int(box.cls[0])
        if cls == 1:
            conf_score = float(box.conf[0])
            falhaDetetada = True
            counter += 1
            print("Falhas identificadas na imagem: ", counter, " - score: ", conf_score)
                        
if results[0].masks is not None:
    print("Máscaras encontradas:", len(results[0].masks.xy))
    
pasta_projeto = r"C:\Users\frrodrigues.INDPT\Desktop\projeto_defeitos_manta\passo_2_teste_modelo"
caminho_saida = os.path.join(pasta_projeto, "img_teste_v15.jpg")
results[0].save(filename=caminho_saida)

