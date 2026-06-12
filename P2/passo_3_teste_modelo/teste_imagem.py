from ultralytics import YOLO
import os


caminho = r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_3_treino_modelo\treino_falhas_lotes\runs\detect\train\weights\best.pt"
model = YOLO(caminho)

results = model(r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_1_extracao_imagens_frames\novas_imagens\v7_datase_falhas_crop\falhas (2).jpg", verbose=False, conf=0.1)

counter = 0
last_detection = 0
falhaDetetada = False


for box in results[0].boxes:
        cls = int(box.cls[0])
        if cls == 0:
            falhaDetetada = True
            counter += 1
            print("Falhas identificadas na imagem: ", counter)
                        

pasta_projeto = r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_4_teste_modelo\detecao_falhas_lotes"
caminho_saida = os.path.join(pasta_projeto, "img_deteaco_falha.jpg")
results[0].save(filename=caminho_saida)
print("Falhas identificadas na imagem: ", counter)

