import cv2
import os

    
video_path = r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_1_extracao_imagens_frames\novas_imagens\novos_videos\Export_2026-05-11_09_38_05_292\P127 – ACB – Formação de lotes – Mesa 2 Trás 2026-05-09_10_21_26_071.mkv"
output_folder = r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_1_extracao_imagens_frames\novas_imagens\frames_extracao"
    
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_count = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 30 == 0:
        filename = f"{output_folder}/falha_{saved}.jpg"
        cv2.imwrite(filename, frame)
        saved += 1

    frame_count += 1

cap.release()

print("Frames extraídos:", saved)