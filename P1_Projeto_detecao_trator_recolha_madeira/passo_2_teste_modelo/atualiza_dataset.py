from ultralytics import YOLO

caminho = r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_3_treino_modelo\treino_melhoria\runs\detect\train\weights\best.pt"
model = YOLO(caminho)
results = model.predict(
    source=r'C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_4_teste_modelo\novas_imagens_v2', 
    save_txt=True,      # Guarda os ficheiros .txt (as labels)
    save_conf=True,     # Guarda a confiança (ajuda a filtrar o que está mal)
    save=True           # Guarda as fotos com as caixas desenhadas para tu veres logo o que ele falhou
)