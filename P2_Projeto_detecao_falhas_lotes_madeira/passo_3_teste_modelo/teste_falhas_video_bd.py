import cv2
from ultralytics import YOLO
from datetime import datetime, timedelta
from supabase import create_client, Client
import io
import threading
import sys
import os

SUPABASE_URL = "https://rqxkyqmbigutmrftjclh.supabase.co"
SUPABASE_KEY = "sb_secret_KEnpyvmtnl-38Vfx6yaqgg_UwDHL9-D"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
bucket_name = "fotos_detecao_falhas"


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

modelo_lotes = YOLO(resource_path("modelo_lotes.pt"))
modelo_falhas = YOLO(resource_path("modelo_falhas_lotes.pt"))

caminho_video = r"C:\Users\frrodrigues\Downloads\P127 – ACB – Formação de lotes – Mesa 2 Trás 2026-04-10_02_11_41_289 - Trim - Trim - Trim.mp4"
def guarda_bd(t_inicio, imagem, imagem_original, area):
    try:
        _, buffer = cv2.imencode('.jpg', imagem, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        file_bits = io.BytesIO(buffer).read()
        file_path = f"foto_{t_inicio.replace(' ', '_').replace(':', '-')}.jpg"
        
        supabase.storage.from_(bucket_name).upload(file_path, file_bits, {"content-type": "image/jpeg"})
        url_publica = supabase.storage.from_(bucket_name).get_public_url(file_path)
        
        
        _, buff_orig = cv2.imencode('.jpg', imagem_original, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        name_orig = f"original_{t_inicio.replace(' ', '_').replace(':', '-')}.jpg"
        supabase.storage.from_(bucket_name).upload(name_orig, io.BytesIO(buff_orig).read(), {"content-type": "image/jpeg", "upsert": "true"})
        url_orig = supabase.storage.from_(bucket_name).get_public_url(name_orig)
        
        data = {"timestamp_inicio": t_inicio, "imagem_detecao": url_publica, "imagem_original": url_orig, "area_falha": area}
        supabase.table("detecoes_falhas").insert(data).execute()
        agora= datetime.now() 
        print("Sucesso: Detecção enviada para Supabase as {agora}" )
    except Exception as e:
        print(f"Erro ao guardar: {e}")



def gen_frames():
    cap = cv2.VideoCapture(caminho_video)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo!")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    delay = int(1000 / fps)

    lote_ja_notificado = False
    frames_sem_lote = 0
    TOLERANCIA_FRAMES = 10
    ultima_detecao = datetime.now() - timedelta(seconds=6)
    paused = False

    while True:
        if not paused:
            success, frame = cap.read()
            if not success:
                break

        results_lotes = modelo_lotes(frame, verbose=False, conf=0.8)
        annotated_frame = results_lotes[0].plot()
        boxes_lote = [box for box in results_lotes[0].boxes if int(box.cls[0]) == 0]

        falha_detetada_neste_frame = False
        img_para_enviar = None
        area_px = None
        imagem_original = None

        agora = datetime.now()
        cooldown_ativo = (agora - ultima_detecao).total_seconds() < 6

        if len(boxes_lote) > 0:
            frames_sem_lote = 0

            if not lote_ja_notificado:
                print("--- Detetei novo lote ---")
                lote_ja_notificado = True

            if not cooldown_ativo:  # só analisa falhas se o cooldown tiver passado
                for box in boxes_lote:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    lote_roi = frame[y1:y2, x1:x2]
                    imagem_original = lote_roi.copy()

                    if lote_roi.size > 0:
                        results_falhas = modelo_falhas(lote_roi, verbose=False, conf=0.5)

                        for box_falha in results_falhas[0].boxes:
                            xf1, yf1, xf2, yf2 = map(int, box_falha.xyxy[0])
                            largura_px = xf2 - xf1
                            altura_px = yf2 - yf1
                            area_px = largura_px * altura_px
                            print(f"Tamanho da falha: {largura_px}x{altura_px} px — area: {area_px} px²")
                            falha_detetada_neste_frame = True
                            img_para_enviar = results_falhas[0].plot()
                            print(">>> Falha detetada!")

        else:
            frames_sem_lote += 1
            if frames_sem_lote > TOLERANCIA_FRAMES:
                lote_ja_notificado = False

        if falha_detetada_neste_frame:
            ja = agora.strftime('%Y-%m-%d %H:%M:%S')
            print(f"Enviando deteção para o Supabase... {ja}")
            ultima_detecao = agora  # inicia o cooldown de 6 segundos
            threading.Thread(target=guarda_bd, args=(ja, img_para_enviar, imagem_original, area_px)).start()
            cv2.imwrite("testee.jpg", img_para_enviar)

        cv2.imshow("Video - Deteção em Cascata", annotated_frame)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'): break
        if key == ord('p'): paused = not paused

  
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    gen_frames()
    