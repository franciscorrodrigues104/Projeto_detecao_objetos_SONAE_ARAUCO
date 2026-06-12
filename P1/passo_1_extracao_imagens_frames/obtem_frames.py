#este ficheiro é responsavel por obter frames de vídeos (caso tenha vídeos e não imagens)
import cv2
import os

    
video_path = r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_1_extracao_imagens_frames\extracao_videos\pt_2\background\background (7).mkv"
output_folder = "frames"
    
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_count = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 60 == 0:
        filename = f"{output_folder}/frame_{saved}.jpg"
        cv2.imwrite(filename, frame)
        saved += 1

    frame_count += 1

cap.release()

print("Frames extraídos:", saved)