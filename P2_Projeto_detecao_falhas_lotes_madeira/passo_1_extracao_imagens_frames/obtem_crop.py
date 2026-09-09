from ultralytics import YOLO

caminho = r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_5_detecao_front_end\app_detecao\modelo_lotes_v2.pt"
model = YOLO(caminho)
results = model.predict(
    source=r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_1_extracao_imagens_frames\novas_imagens\frames_extracao", 
    save_crop=True,
    save_txt=True,      # Guarda os ficheiros .txt (as labels)
    save_conf=True,     # Guarda a confiança (ajuda a filtrar o que está mal)
    save=False           # Guarda as fotos com as caixas desenhadas para tu veres logo o que ele falhou
)