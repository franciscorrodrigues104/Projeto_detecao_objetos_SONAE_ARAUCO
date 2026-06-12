import cv2
from ultralytics import YOLO
import time
from datetime import datetime
import os

# --- CONFIGURAÇÕES DE CAMINHOS ---
caminho_modelo = r"C:\Users\frrodrigues\Desktop\Projeto_detecao_trator\passo_4_teste_modelo\best.pt"
caminho_video = r"C:\Users\frrodrigues\Downloads\P103 – PAL – Rejeitados Grossos e não ferrosos 2026-03-13_02_57_57_423 - Trim.mp4"
pasta_prints = r"C:\Users\frrodrigues\Desktop\Projeto_detecao_trator\passo_4_teste_modelo\imagens_detecao_video"
caminho_video_saida = r"C:\Users\frrodrigues\Desktop\Projeto_detecao_trator\passo_4_teste_modelo\resultado_deteccao.mp4"

# Criar pasta de prints se não existir
if not os.path.exists(pasta_prints):
    os.makedirs(pasta_prints)

# --- INICIALIZAÇÃO ---
model = YOLO(caminho_modelo)
cap = cv2.VideoCapture(caminho_video)

if not cap.isOpened():
    print("Erro ao abrir o vídeo!")
    exit()

# Propriedades do vídeo para gravação
largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0: fps = 30

# Configuração do VideoWriter (Gravação)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(caminho_video_saida, fourcc, fps, (largura, altura))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cv2.namedWindow("Video")

def mudar_frame(x):
    cap.set(cv2.CAP_PROP_POS_FRAMES, x)

cv2.createTrackbar("Frame", "Video", 0, total_frames, mudar_frame)

# --- VARIÁVEIS DE CONTROLO ---
paused = False
ultimo_frame_com_trator = None
tempo_total_trator = 0.0
trator_estava_presente = False
ultima_deteccao_tempo = 0 
timestamp_inicio = ""

print("Comandos: 'p' para Pausa/Play, 'q' para Sair.")

# --- CICLO PRINCIPAL ---
while True:
    trator_detetado_neste_frame = False 
    
    if not paused:
        success, frame = cap.read()
        if not success:
            break

        # Atualizar posição da trackbar
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.setTrackbarPos("Frame", "Video", current_frame)

        # Inferência com YOLO
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        tempo_atual = time.time()

        # Gravar o frame no vídeo de saída
        out.write(annotated_frame)

        # Lógica de Deteção de Classe
        for box in results[0].boxes:
            if int(box.cls[0]) == 0: # Ajusta o ID da classe se necessário
                trator_detetado_neste_frame = True
                break 

        if trator_detetado_neste_frame:
            if ultimo_frame_com_trator is not None:
                delta = tempo_atual - ultimo_frame_com_trator
                tempo_total_trator += delta
            
            ultimo_frame_com_trator = tempo_atual
            
            if not trator_estava_presente:
                timestamp_inicio = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                trator_estava_presente = True
            
            # Guardar Print de 5 em 5 segundos enquanto deteta
            if tempo_atual - ultima_deteccao_tempo > 5:
                agora = datetime.now()
                nome_arquivo = agora.strftime('deteccao_%Y%m%d_%H%M%S.jpg')
                caminho_save = os.path.join(pasta_prints, nome_arquivo)
                cv2.imwrite(caminho_save, annotated_frame)
                print(f"Print salva em: {nome_arquivo}")
                ultima_deteccao_tempo = tempo_atual

        else:
            # Se o objeto desapareceu do frame
            if trator_estava_presente:
                timestamp_fim = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"\n--- DETEÇÃO FINALIZADA ---")
                print(f"Início: {timestamp_inicio}")
                print(f"Fim:    {timestamp_fim}")
                print(f"Duração da falha: {tempo_total_trator:.2f} segundos.")
                
                # Reset de variáveis
                trator_estava_presente = False
                tempo_total_trator = 0.0 
                ultimo_frame_com_trator = None

        cv2.imshow("Video", annotated_frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'): 
        break
    if key == ord('p'): 
        paused = not paused

# --- FINALIZAÇÃO ---
print("\nA processar fecho dos ficheiros... aguarde.")
cap.release()
out.release() # Fecha o vídeo gravado
cv2.destroyAllWindows()
print(f"Vídeo gravado com sucesso em: {caminho_video_saida}")