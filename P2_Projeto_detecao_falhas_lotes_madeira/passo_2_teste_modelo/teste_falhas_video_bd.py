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

# --- Parâmetros da janela de deteção por lote ---
JANELA_DURACAO = 120        # duração da janela, em segundos (2 minutos)
CONFIANCA_MINIMA = 0.65     # confidence mínimo (exclusivo) para inserir na BD


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

modelo_lotes = YOLO(resource_path("modelo_lotes.pt"))
modelo_falhas = YOLO(resource_path("modelo_falhas_lotes.pt"))

# --- Filtra, à partida, quais as classes do modelo de falhas que interessam ---
# Ajusta os nomes abaixo caso não correspondam exatamente aos nomes usados no teu modelo
# (podes confirmar correndo: print(modelo_falhas.names))
nomes_classes_falhas = modelo_falhas.names
CLASSES_RELEVANTES_NOMES = {"desalinhamento", "falhas"}
CLASSES_RELEVANTES_IDS = [k for k, v in nomes_classes_falhas.items() if v in CLASSES_RELEVANTES_NOMES]

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
        agora = datetime.now()
        print(f"Sucesso: Detecção enviada para Supabase as {agora}")
    except Exception as e:
        print(f"Erro ao guardar: {e}")


def finaliza_janela(janela_deteccoes):
    """Avalia as deteções acumuladas numa janela e insere na BD apenas a de maior
    confidence, caso ultrapasse CONFIANCA_MINIMA."""
    if not janela_deteccoes:
        print("Janela fechada: sem deteções relevantes")
        return

    melhor = max(janela_deteccoes, key=lambda d: d["confianca"])

    if melhor["confianca"] > CONFIANCA_MINIMA:
        print(f"Janela fechada: melhor confidence {melhor['confianca']:.2f} ({melhor['classe']}) -> a enviar para Supabase")
        threading.Thread(
            target=guarda_bd,
            args=(melhor["timestamp"], melhor["imagem"], melhor["imagem_original"], melhor["area"])
        ).start()
        cv2.imwrite("testee.jpg", melhor["imagem"])
    else:
        print(f"Janela fechada: melhor confidence {melhor['confianca']:.2f} ({melhor['classe']}), abaixo de {CONFIANCA_MINIMA}, nada inserido")


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
    paused = False

    # --- Estado da janela de deteção (1 janela = 1 lote a passar) ---
    janela_deteccoes = []
    janela_ativa = False
    janela_inicio = None

    while True:
        if not paused:
            success, frame = cap.read()
            if not success:
                break

        results_lotes = modelo_lotes(frame, verbose=False, conf=0.8)
        annotated_frame = results_lotes[0].plot()
        boxes_lote = [box for box in results_lotes[0].boxes if int(box.cls[0]) == 0]

        houve_lote_neste_frame = len(boxes_lote) > 0
        agora = datetime.now()

        # --- Fecha a janela ao fim de 2 minutos ---
        if janela_ativa and janela_inicio and (agora - janela_inicio).total_seconds() >= JANELA_DURACAO:
            finaliza_janela(janela_deteccoes)
            janela_deteccoes = []
            janela_ativa = False
            janela_inicio = None

        # --- Abre nova janela apenas se não houver nenhuma ativa e aparecer um lote ---
        if not janela_ativa and houve_lote_neste_frame:
            janela_ativa = True
            janela_inicio = agora
            janela_deteccoes = []
            print("--- Nova janela de deteção iniciada ---")

        if houve_lote_neste_frame:
            frames_sem_lote = 0

            if not lote_ja_notificado:
                print("--- Detetei novo lote ---")
                lote_ja_notificado = True

            if janela_ativa:
                for box in boxes_lote:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    lote_roi = frame[y1:y2, x1:x2]
                    if lote_roi.size == 0:
                        continue

                    imagem_original = lote_roi.copy()

                    results_falhas = modelo_falhas(
                        lote_roi, verbose=False, conf=0.5, classes=CLASSES_RELEVANTES_IDS
                    )
                    nomes_classes = results_falhas[0].names

                    for box_falha in results_falhas[0].boxes:
                        classe_id = int(box_falha.cls[0])
                        nome_classe = nomes_classes[classe_id]

                        confianca = float(box_falha.conf[0])
                        xf1, yf1, xf2, yf2 = map(int, box_falha.xyxy[0])
                        largura_px = xf2 - xf1
                        altura_px = yf2 - yf1
                        area_px = largura_px * altura_px

                        print(f"Falha detetada - classe: {nome_classe}, confidence: {confianca:.2f}, tamanho: {largura_px}x{altura_px} px — area: {area_px} px²")

                        janela_deteccoes.append({
                            "confianca": confianca,
                            "classe": nome_classe,
                            "area": area_px,
                            "imagem": results_falhas[0].plot(),
                            "imagem_original": imagem_original,
                            "timestamp": agora.strftime('%Y-%m-%d %H:%M:%S'),
                        })

        else:
            frames_sem_lote += 1
            if frames_sem_lote > TOLERANCIA_FRAMES:
                lote_ja_notificado = False

        cv2.imshow("Video - Deteção em Cascata", annotated_frame)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            paused = not paused

    # Se o vídeo terminar com uma janela ainda ativa, avalia-a antes de sair
    if janela_ativa:
        finaliza_janela(janela_deteccoes)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    gen_frames()