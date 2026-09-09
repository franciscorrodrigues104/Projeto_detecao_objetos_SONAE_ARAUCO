import cv2
import os

    
video_path = r"C:\Users\frrodrigues.INDPT\Desktop\projeto_defeitos_manta\passo_1_extracao_imagens\Export_2026-09-03_11_29_53_344\P068 – LC – Manta de Formação 2026-09-03_11_14_12_145.mkv"
output_folder = r"C:\Users\frrodrigues.INDPT\Desktop\projeto_defeitos_manta\passo_1_extracao_imagens\img_teste"
    
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