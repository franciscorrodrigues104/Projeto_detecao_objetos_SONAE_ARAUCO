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
            
        if frame_count % 30 == 0:
            # 3. Nome único baseado no nome do vídeo + contador
            filename = f"{video_name_no_ext}_frame_{saved:05d}.jpg"
            path_completo = os.path.join(output_folder, filename)
            
            cv2.imwrite(path_completo, frame)
            saved += 1
            
        frame_count += 1
        
    cap.release()
    print(f"Extraídos {saved} frames do vídeo: {video_name_no_ext}")

# --- Como usar para múltiplos vídeos ---
output_folder = r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_1_extracao_imagens_frames\pt4_melhoria\frames"
os.makedirs(output_folder, exist_ok=True)

# Lista de vídeos que queres processar
lista_videos = [
r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_1_extracao_imagens_frames\pt4_melhoria\Export_2026-08-19_17_02_05_104\P125 – ACB – Formação de lotes – Mesa 1 Trás 2026-08-19_13_12_02_644.mkv"
,r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_1_extracao_imagens_frames\pt4_melhoria\Export_2026-08-19_17_02_05_104\P125 – ACB – Formação de lotes – Mesa 1 Trás 2026-08-19_13_21_06_496.mkv"
,r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_1_extracao_imagens_frames\pt4_melhoria\Export_2026-08-19_17_02_05_104\P125 – ACB – Formação de lotes – Mesa 1 Trás 2026-08-19_13_39_45_753.mkv"
,r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_1_extracao_imagens_frames\pt4_melhoria\Export_2026-08-19_17_02_05_104\P125 – ACB – Formação de lotes – Mesa 1 Trás 2026-08-19_13_48_48_349.mkv"
,r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_1_extracao_imagens_frames\pt4_melhoria\Export_2026-08-19_17_47_17_471\P125 – ACB – Formação de lotes – Mesa 1 Trás 2026-08-18_14_29_36_654.mkv"
,r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_1_extracao_imagens_frames\pt4_melhoria\Export_2026-08-19_17_47_17_471\P125 – ACB – Formação de lotes – Mesa 1 Trás 2026-08-18_23_59_02_120.mkv"
,r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_1_extracao_imagens_frames\pt4_melhoria\Export_2026-08-19_17_47_17_471\P125 – ACB – Formação de lotes – Mesa 1 Trás 2026-08-19_00_27_35_434.mkv"
,r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_1_extracao_imagens_frames\pt4_melhoria\Export_2026-08-19_17_47_17_471\P125 – ACB – Formação de lotes – Mesa 1 Trás 2026-08-19_01_13_48_415.mkv"
,r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_1_extracao_imagens_frames\pt4_melhoria\Export_2026-08-19_17_47_17_471\P125 – ACB – Formação de lotes – Mesa 1 Trás 2026-08-19_03_26_47_587.mkv"
,r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_1_extracao_imagens_frames\pt4_melhoria\Export_2026-08-19_17_47_17_471\P125 – ACB – Formação de lotes – Mesa 1 Trás 2026-08-19_03_30_22_777.mkv"
    ]

for video in lista_videos:
    extrair_frames(video, output_folder)