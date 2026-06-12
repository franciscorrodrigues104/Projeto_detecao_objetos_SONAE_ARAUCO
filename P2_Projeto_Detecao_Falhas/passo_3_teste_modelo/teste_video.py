import cv2
from ultralytics import YOLO
import time
from datetime import datetime
import time
import os


ultimo_frame_com_trator = None
tempo_total_trator = 0.0

caminho_modelo = r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_3_treino_modelo\treino_lotes\runs\detect\train\weights\best.pt"
model = YOLO(caminho_modelo)

caminho_video = r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_1_extracao_imagens_frames\novas_imagens\Export_2026-04-20_10_56_18_835\P127 – ACB – Formação de lotes – Mesa 2 Trás 2026-04-20_06_13_10_719.mkv"
cap = cv2.VideoCapture(caminho_video)

if not cap.isOpened():
    print("Erro ao abrir o vídeo!")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cv2.namedWindow("Video")


def mudar_frame(x):
    cap.set(cv2.CAP_PROP_POS_FRAMES, x)

cv2.createTrackbar("Frame", "Video", 0, total_frames, mudar_frame)

paused = False
print("Comandos: 'p' para Pausa/Play, 'q' para Sair.")

ultima_deteccao_tempo = 0 


ultimo_frame_com_trator = None
tempo_total_trator = 0.0
trator_estava_presente = False

while True:
    trator_detetado_neste_frame = False 
    
    if not paused:
        success, frame = cap.read()
        if not success: break

    if success:
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        tempo_atual = time.time()

        for box in results[0].boxes:
            if int(box.cls[0]) == 0:
                trator_detetado_neste_frame = True
                break 

        if trator_detetado_neste_frame:
            if ultimo_frame_com_trator is not None:
                delta = tempo_atual - ultimo_frame_com_trator
                
                tempo_total_trator += delta
            
            ultimo_frame_com_trator = tempo_atual
            trator_estava_presente = True 
            
            if tempo_atual - ultima_deteccao_tempo > 5:
                agora = datetime.now()
                timestamp_inicio = agora.strftime('%Y-%m-%d %H:%M:%S')
                
                ultima_deteccao_tempo = tempo_atual
                pasta_projeto = r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_4_teste_modelo\imagens_detecao_lotes"

                nome_arquivo = agora.strftime('deteccao_%Y%m%d_%H%M%S.jpg')
                caminho_save = os.path.join(pasta_projeto, nome_arquivo)
                
                cv2.imwrite(caminho_save, annotated_frame)
                print(f"Print salva em: {nome_arquivo}")

        else:
           
            if trator_estava_presente:
                if ultimo_frame_com_trator:
                    agora = datetime.now()
                    timestamp_fim = agora.strftime('%Y-%m-%d %H:%M:%S')
                    print(f"DETEÇÃO FINALIZADA!")
                    print(f"Inicio da deteção: {timestamp_inicio}")

                    print(f"Fim da deteção: {timestamp_fim}")
                    
                    print(f"A falha esteve presente por: {tempo_total_trator:.2f} segundos.")
                    
                    trator_estava_presente = False
                    tempo_total_trator = 0.0 
                    ultimo_frame_com_trator = None

        cv2.imshow("Video", annotated_frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'): break
    if key == ord('p'): paused = not paused






cap.release()
cv2.destroyAllWindows()
