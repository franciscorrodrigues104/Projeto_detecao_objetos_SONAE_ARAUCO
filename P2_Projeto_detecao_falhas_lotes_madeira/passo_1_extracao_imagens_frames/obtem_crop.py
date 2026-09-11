from ultralytics import YOLO

caminho = r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_4_detecao_front_end\app_detecao\modelo_lotes_v4.pt"
model = YOLO(caminho)
results = model.predict(
    source=r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_1_extracao_imagens_frames\pt4_melhoria\frames", 
    save_crop=True,
    save_txt=True,      # Guarda os ficheiros .txt (as labels)
    save_conf=True,     # Guarda a confiança (ajuda a filtrar o que está mal)
    save=False           # Guarda as fotos com as caixas desenhadas para tu veres logo o que ele falhou
)