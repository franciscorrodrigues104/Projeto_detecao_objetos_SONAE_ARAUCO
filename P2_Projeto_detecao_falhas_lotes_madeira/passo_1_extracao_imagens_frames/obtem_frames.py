import cv2
import os

    
video_path = r"C:\Users\frrodrigues\Desktop\frames_dataset_v2_falhas\Export_2026-07-14_11_11_09_668\P127 – ACB – Formação de lotes – Mesa 2 Trás 2026-07-06_12_26_13_920.mkv"

output_folder = r"C:\Users\frrodrigues\Desktop\frames_dataset_v2_falhas\dataset_novo_frames"
    
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