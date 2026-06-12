from ultralytics import YOLO
import cv2

# 1. Carregar os dois modelos
model_lotes = YOLO(r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_3_treino_modelo\treino_lotes\runs\detect\train\weights\last.pt")
model_falhas = YOLO(r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_4_teste_modelo\modelo_falhas_lotes.pt") # O seu modelo de falhas

# 2. Carregar imagem
img = cv2.imread(r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_1_extracao_imagens_frames\novas_imagens\v8\falha (22).jpg")

# 3. Localizar o lote
res_lote = model_lotes(img)

for box in res_lote[0].boxes:
    # Obter coordenadas da caixa do lote
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    # 4. Criar o recorte (ROI - Region of Interest)
    roi = img[y1:y2, x1:x2]
    
    # 5. Inspecionar o recorte em busca de falhas
    res_falhas = model_falhas(roi, conf = 0.4)
    
    # 6. Analisar resultados das falhas
    if len(res_falhas[0].boxes) > 0:
        
        print(f"ALERTA: {len(res_falhas[0].boxes)} falha(s) detetada(s) no lote!")
        # Opcional: guardar a imagem do recorte com a falha detetada
        res_falhas[0].save(filename="falha_detetada_t.jpg")