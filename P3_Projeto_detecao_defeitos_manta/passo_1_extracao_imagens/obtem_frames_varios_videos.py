import cv2
import os

def extrair_frames(video_path, output_folder):
    # 1. Obter apenas o nome do ficheiro (ex: 'video_lote_A.mkv')
    video_filename = os.path.basename(video_path)
    # 2. Remover a extensão para usar como prefixo (ex: 'video_lote_A')
    video_name_no_ext = os.path.splitext(video_filename)[0]
    
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    saved = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % 10 == 0:
            # 3. Nome único baseado no nome do vídeo + contador
            filename = f"{video_name_no_ext}_frame_{saved:05d}.jpg"
            path_completo = os.path.join(output_folder, filename)
            
            cv2.imwrite(path_completo, frame)
            saved += 1
            
        frame_count += 1
        
    cap.release()
    print(f"Extraídos {saved} frames do vídeo: {video_name_no_ext}")

# --- Como usar para múltiplos vídeos ---
output_folder = r"C:\Users\frrodrigues.INDPT\Desktop\projeto_defeitos_manta\passo_1_extracao_imagens\novos_frames_defeitos"
os.makedirs(output_folder, exist_ok=True)

# Lista de vídeos que queres processar
lista_videos = [
r"C:\Users\frrodrigues.INDPT\Desktop\projeto_defeitos_manta\passo_1_extracao_imagens\Export_2026-09-07_15_02_52_140\P068 – LC – Manta de Formação 2026-09-02_08_09_29_979.mkv"
,r"C:\Users\frrodrigues.INDPT\Desktop\projeto_defeitos_manta\passo_1_extracao_imagens\Export_2026-09-07_15_02_52_140\P068 – LC – Manta de Formação 2026-09-02_08_19_55_439.mkv"
,r"C:\Users\frrodrigues.INDPT\Desktop\projeto_defeitos_manta\passo_1_extracao_imagens\Export_2026-09-07_15_02_52_140\P068 – LC – Manta de Formação 2026-09-02_08_50_21_477.mkv"
,r"C:\Users\frrodrigues.INDPT\Desktop\projeto_defeitos_manta\passo_1_extracao_imagens\Export_2026-09-07_15_02_52_140\P068 – LC – Manta de Formação 2026-09-02_08_51_26_298.mkv"
,r"C:\Users\frrodrigues.INDPT\Desktop\projeto_defeitos_manta\passo_1_extracao_imagens\Export_2026-09-07_15_02_52_140\P068 – LC – Manta de Formação 2026-09-03_00_04_36_872.mkv"
,r"C:\Users\frrodrigues.INDPT\Desktop\projeto_defeitos_manta\passo_1_extracao_imagens\Export_2026-09-07_15_02_52_140\P068 – LC – Manta de Formação 2026-09-03_00_07_04_975.mkv"
,r"C:\Users\frrodrigues.INDPT\Desktop\projeto_defeitos_manta\passo_1_extracao_imagens\Export_2026-09-07_15_02_52_140\P068 – LC – Manta de Formação 2026-09-03_00_10_41_060.mkv"
,r"C:\Users\frrodrigues.INDPT\Desktop\projeto_defeitos_manta\passo_1_extracao_imagens\Export_2026-09-07_15_02_52_140\P068 – LC – Manta de Formação 2026-09-03_00_15_54_242.mkv"
    ]

for video in lista_videos:
    extrair_frames(video, output_folder)