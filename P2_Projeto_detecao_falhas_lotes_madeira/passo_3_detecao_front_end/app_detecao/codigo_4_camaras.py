import cv2
from flask import Flask, Response, render_template, redirect, request
from ultralytics import YOLO
import time
from datetime import datetime, timedelta
import os
import webbrowser
from threading import Timer
import sys
from supabase import create_client, Client
import io
import threading
from dotenv import load_dotenv
import logging
import copy
import torch

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
bucket_name = "bucket_falhas_v2"


log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
log_file = open("logs.txt", "a", encoding="utf-8", buffering=1)
sys.stdout = log_file
sys.stderr = log_file


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# --------------------------------------------------------------------------
# CONFIGURAÇÃO DAS CÂMARAS
# Adiciona ou remove entradas aqui consoante o número de câmaras que tens.
# Cada câmara precisa de uma variável de ambiente própria no .env, ex:
#   CAMERA_URL_1=rtsp://...
#   CAMERA_URL_2=rtsp://...
#   CAMERA_URL_3=rtsp://...
# --------------------------------------------------------------------------
CAMERAS = [
     {"id": 1, "nome": "Mesa 1 - Trás", "url": os.getenv("CAMERA_URL_1")},
    {"id": 2, "nome": "Mesa 1 - Frente", "url": os.getenv("CAMERA_URL_2")},
    {"id": 3, "nome": "Mesa 2 - Trás", "url": os.getenv("CAMERA_URL_3")},
    {"id": 4, "nome": "Mesa 2 - Frente", "url": os.getenv("CAMERA_URL_4")},
]

modelo_lotes = YOLO(resource_path("modelo_lotes_v4.pt"))
modelo_falhas = YOLO(resource_path("modelo_falhas_v10.pt"))

# Locks para proteger cada modelo partilhado de chamadas concorrentes
# vindas de threads de câmaras diferentes.
lock_modelo_lotes = threading.Lock()
lock_modelo_falhas = threading.Lock()

app = Flask(__name__, template_folder=resource_path("templates"), static_folder=resource_path("static"))

insercao_lock = threading.Lock()

# Estado partilhado por câmara:
#   "frame"          -> último frame capturado, pronto a mostrar no /video_feed
#   "frame_deteccao" -> último frame capturado, usado pela thread de inferência
# Separar os dois é o que permite que a exibição nunca fique à espera do YOLO.
camera_state = {
    cam["id"]: {"lock": threading.Lock(), "frame": None, "frame_deteccao": None}
    for cam in CAMERAS
}


# Classes cujas caixas devem aparecer desenhadas na imagem de deteção
# gravada. Todas as classes continuam a ser inferidas na mesma (usadas na
# lógica de deteção); isto só controla o que o YOLO desenha na imagem.
CLASSES_VISIVEIS_NA_IMAGEM = {"desalinhamento", "falhas"}


def desenhar_falhas_visiveis(results_falhas):
    """
    Usa o desenho nativo do YOLO (results.plot(), com o estilo automático de
    caixas/labels/cores), mas filtra antes as caixas para conter apenas as
    classes em CLASSES_VISIVEIS_NA_IMAGEM. As restantes classes detetadas
    (ex: alinhado, bordo_saudavel) continuam a ser inferidas normalmente,
    só não aparecem desenhadas.
    """
    resultado = results_falhas[0]
    nomes_classes = resultado.names

    mask = torch.tensor(
        [nomes_classes[int(c)] in CLASSES_VISIVEIS_NA_IMAGEM for c in resultado.boxes.cls],
        dtype=torch.bool,
    )

    resultado_filtrado = copy.copy(resultado)
    resultado_filtrado.boxes = resultado.boxes[mask]

    return resultado_filtrado.plot()


def guarda_bd(t_inicio, imagem, imagem_original, area, camera_id, camera_nome):
    with insercao_lock:
        try:
            _, buffer = cv2.imencode('.jpg', imagem, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            file_bits = io.BytesIO(buffer).read()

            ts_seguro = t_inicio.replace(' ', '_').replace(':', '-')
            nome_deteccao = f"cam{camera_id}_foto_{ts_seguro}.jpg"

            supabase.storage.from_(bucket_name).upload(
                nome_deteccao,
                file_bits,
                {"content-type": "image/jpeg", "upsert": "true"}
            )

            _, buff_orig = cv2.imencode('.jpg', imagem_original, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            nome_original = f"cam{camera_id}_original_{ts_seguro}.jpg"
            supabase.storage.from_(bucket_name).upload(nome_original, io.BytesIO(buff_orig).read(), {"content-type": "image/jpeg", "upsert": "true"})
            url_orig = supabase.storage.from_(bucket_name).get_public_url(nome_original)

            url_publica = supabase.storage.from_(bucket_name).get_public_url(nome_deteccao)

            data_hoje = datetime.now().strftime('%Y-%m-%d')
            response_count = supabase.table("detecoes_falhas_v2")\
                .select("*", count="exact")\
                .filter("timestamp_inicio", "ilike", f"{data_hoje}%")\
                .execute()

            proximo_contador = (response_count.count or 0) + 1

            data = {
                "timestamp_inicio": t_inicio,
                "counter_dia": proximo_contador,
                "imagem_detecao": url_publica,
                "imagem_original": url_orig,
                "area_falha": area,
                "nome_camara": camera_nome,
            }

            supabase.table("detecoes_falhas_v2").insert(data).execute()
            print(f"[{camera_nome}] Inserido com sucesso: (Contador: {proximo_contador})")

        except Exception as e:
            print(f"[{camera_nome}] Erro ao guardar na base de dados: {e}")


def camera_capture_loop(camera):
    """
    Loop de captura para UMA câmara. Só lê frames da câmara e atualiza o
    estado partilhado — nunca espera pelo YOLO. É isto que garante que o
    vídeo mostrado no browser não fica em atraso/"congelado", independente
    de quão ocupada a inferência estiver com as outras câmaras.
    """
    camera_id = camera["id"]
    camera_nome = camera["nome"]
    camera_url = camera["url"]
    estado = camera_state[camera_id]

    if not camera_url:
        print(f"[{camera_nome}] URL não configurado (verifica o .env). Thread não iniciada.")
        return

    cap = cv2.VideoCapture(camera_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        if not cap.isOpened():
            print(f"[{camera_nome}] Câmara desligada, a tentar reconectar...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(camera_url)
            continue

        success, frame = cap.read()
        if not success:
            print(f"[{camera_nome}] Falha ao ler frame, a reiniciar stream...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(camera_url)
            continue


        with estado["lock"]:
            estado["frame"] = cv2.resize(frame, (800, 450))  # para o /video_feed
            estado["frame_deteccao"] = frame                  # para a thread de inferência


def inference_loop(camera):
    """
    Loop de inferência para UMA câmara. Corre em paralelo com a captura,
    pegando sempre no frame mais recente disponível e correndo os dois
    modelos YOLO sobre ele.

    Em vez de gravar uma deteção por cada frame em que uma falha aparece
    (o que geraria várias entradas repetidas para o mesmo lote físico),
    guarda-se apenas a deteção com maior confiança enquanto o lote está a
    passar. Só quando o lote deixa de ser visto durante
    TEMPO_SEM_LOTE_PARA_FINALIZAR segundos (consideramos que já saiu da
    câmara) é que essa melhor deteção é gravada na base de dados.

    Exceção: se aparecer uma deteção com confiança acima de
    CONFIANCA_INSERCAO_IMEDIATA, é claramente uma falha e é gravada de
    imediato, sem esperar pelo fim do lote. Depois disso, ignoram-se as
    restantes deteções desse mesmo lote (já não interessa continuar a
    acumular "melhor deteção" nem gravar outra vez quando ele desaparecer).
    """
    camera_id = camera["id"]
    camera_nome = camera["nome"]
    camera_url = camera["url"]
    estado = camera_state[camera_id]

    if not camera_url:
        return

    TEMPO_SEM_LOTE_PARA_FINALIZAR = int(os.getenv("TEMPO_SEM_LOTE_PARA_FINALIZAR", 30))

    CONFIANCA_INSERCAO_IMEDIATA = float(os.getenv("CONFIANCA_INSERCAO_IMEDIATA", 0.72))

    CONFIANCA_FALHAS = float(os.getenv("CONFIANCA_FALHAS", 0.55))
    
    CONFIANCA_LOTES = float(os.getenv("CONFIANCA_LOTES", 0.70))
    
    CLASSES_INSERCAO_IMEDIATA = {"desalinhamento", "falhas"}
    
    CONFIANCA_MINIMA_INSERCAO = float(os.getenv("CONFIANCA_MINIMA_INSERCAO", 0.55))

    lote_ja_anunciado = False
    ultima_vez_lote_visto = None
    melhor_deteccao = None  # dict: conf, area_px, img_para_enviar, imagem_original, timestamp
    insercao_imediata_feita = False  # já gravámos uma deteção de alta confiança para este lote?

    while True:
        with estado["lock"]:
            frame = estado["frame_deteccao"]

        if frame is None:
            time.sleep(0.1)
            continue

        agora = datetime.now()

        with lock_modelo_lotes:
            results_lotes = modelo_lotes(frame, verbose=False, conf=CONFIANCA_LOTES)
        boxes_lote = [box for box in results_lotes[0].boxes if int(box.cls[0]) == 0]

        if len(boxes_lote) > 0:
            print(f"[{camera_nome}] lote presente "f"{agora.strftime('%H:%M:%S')}")
            ultima_vez_lote_visto = agora

            if not lote_ja_anunciado:
                print(f"[{camera_nome}] --- Novo Lote na Linha --- {agora.strftime('%Y-%m-%d %H:%M:%S')}")
                lote_ja_anunciado = True

            for box in boxes_lote:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                lote_roi = frame[y1:y2, x1:x2]

                if lote_roi.size == 0:
                    continue

                with lock_modelo_falhas:
                    results_falhas = modelo_falhas(lote_roi, verbose=False, conf= CONFIANCA_FALHAS, imgsz=800)
                nomes_classes = results_falhas[0].names  # dict {id: nome_da_classe}

                for box_falha in results_falhas[0].boxes:
                    classe_id = int(box_falha.cls[0])
                    nome_classe = nomes_classes[classe_id]

                    if nome_classe == "alinhado" or nome_classe == "bordo_saudavel":
                        continue  # não é falha, ignora

                    # Já gravámos uma deteção de alta confiança para este
                    # lote — ignora tudo o resto até ele sair de vista.
                    if insercao_imediata_feita:
                        continue

                    conf = float(box_falha.conf[0])
                    xf1, yf1, xf2, yf2 = map(int, box_falha.xyxy[0])
                    largura_px = xf2 - xf1
                    altura_px = yf2 - yf1
                    area_px = largura_px * altura_px

                    if nome_classe in CLASSES_INSERCAO_IMEDIATA and conf > CONFIANCA_INSERCAO_IMEDIATA:
                        # Deteção claramente confiante: grava já, sem
                        # esperar os 30s de fim de lote.
                        img_para_enviar = desenhar_falhas_visiveis(results_falhas)
                        imagem_original = lote_roi.copy()

                        threading.Thread(
                            target=guarda_bd,
                            args=(
                                agora.strftime('%Y-%m-%d %H:%M:%S'),
                                img_para_enviar,
                                imagem_original,
                                area_px,
                                camera_id,
                                camera_nome,
                            )
                        ).start()
                        print(f"[{camera_nome}] Deteção de alta confiança (conf={conf:.2f}) — gravada imediatamente - {agora.strftime('%Y-%m-%d %H:%M:%S')}")

                        insercao_imediata_feita = True
                        melhor_deteccao = None  # já não é preciso guardar "melhor" para o fim do lote
                        continue

                    # Confiança abaixo do limiar: continua a acumular a
                    # melhor deteção, a gravar só quando o lote terminar.
                    if melhor_deteccao is None or conf > melhor_deteccao["conf"]:
                        melhor_deteccao = {
                            "conf": conf,
                            "area_px": area_px,
                            "img_para_enviar": desenhar_falhas_visiveis(results_falhas),
                            "imagem_original": lote_roi.copy(),
                            "timestamp": agora,
                        }
                        print(f"[{camera_nome}] Nova melhor deteção do lote atual: conf={conf:.2f}, area={area_px}, {agora.strftime('%Y-%m-%d %H:%M:%S')}")

        else:
            # Não se vê nenhum lote neste frame. Se passou tempo suficiente
            # sem voltar a ver um lote, consideramos que já saiu da câmara.
            if ultima_vez_lote_visto is not None:
                tempo_sem_ver = (agora - ultima_vez_lote_visto).total_seconds()
                print(f"[{camera_nome}] sem lote há "f"{tempo_sem_ver:.1f}s")
                if tempo_sem_ver >= TEMPO_SEM_LOTE_PARA_FINALIZAR:
                    if not insercao_imediata_feita and melhor_deteccao is not None:
                        m = melhor_deteccao
                        if m["conf"] > CONFIANCA_MINIMA_INSERCAO:
                            threading.Thread(
                                target=guarda_bd,
                                args=(
                                    m["timestamp"].strftime('%Y-%m-%d %H:%M:%S'),
                                    m["img_para_enviar"],
                                    m["imagem_original"],
                                    m["area_px"],
                                    camera_id,
                                    camera_nome,
                                )
                            ).start()
                            print(f"[{camera_nome}] Lote terminado, guardada melhor deteção (conf={m['conf']:.2f}) - {agora.strftime('%Y-%m-%d %H:%M:%S')}")
                        else:
                            print(f"[{camera_nome}] Lote terminado, mas confiança da melhor deteção (conf={m['conf']:.2f}) abaixo do limiar mínimo ({CONFIANCA_MINIMA_INSERCAO}) — não foi guardada - {agora.strftime('%Y-%m-%d %H:%M:%S')}")

                    # Reinicia o estado para o próximo lote.
                    melhor_deteccao = None
                    insercao_imediata_feita = False
                    lote_ja_anunciado = False
                    ultima_vez_lote_visto = None

        # Pequena pausa para não martelar o lock dos modelos sem necessidade.
        # Ajusta este valor se precisares de deteção mais rápida (ao custo de mais CPU).
        time.sleep(0.03)


# Inicia, por cada câmara: 1 thread de captura + 1 thread de inferência,
# para que a exibição e a deteção corram totalmente em paralelo.
for cam in CAMERAS:
    threading.Thread(target=camera_capture_loop, args=(cam,), daemon=True).start()
    threading.Thread(target=inference_loop, args=(cam,), daemon=True).start()


@app.route('/', methods=['GET'])
def index():
    data_filtro = request.args.get('data')
    hoje_cpu = datetime.now().strftime('%Y-%m-%d')

    try:
        query = supabase.table("detecoes_falhas_v2").select("*")

        if data_filtro:
            response = query.filter("timestamp_inicio", "ilike", f"{data_filtro}%")\
                            .order("counter_dia", desc=True).execute()
            dados = response.data
            total_real = len(dados)
        else:
            response = query.filter("timestamp_inicio", "ilike", f"{hoje_cpu}%")\
                            .order("counter_dia", desc=True).execute()
            dados = response.data
            total_real = len(dados)

            dados = dados[:20]

    except Exception as e:
        print(f"Erro ao processar dados: {e}")
        dados = []
        total_real = 0

    return render_template('index.html',
                           detecoes=dados,
                           data_selecionada=data_filtro or hoje_cpu,
                           total=total_real,
                           cameras=CAMERAS)


@app.route('/tabela_atualizada', methods=['GET'])
def tabela_atualizada():
    data_filtro = request.args.get('data')
    hoje_cpu = datetime.now().strftime('%Y-%m-%d')
    alvo = data_filtro if data_filtro else hoje_cpu

    try:
        response = supabase.table("detecoes_falhas_v2")\
            .select("*")\
            .filter("timestamp_inicio", "ilike", f"{alvo}%")\
            .order("counter_dia", desc=True).execute()

        dados = response.data
        total_real = len(dados)
        exibir_dados = dados
    except Exception as e:
        print(f"Erro: {e}")
        exibir_dados = []
        total_real = 0

    return render_template('tabela_parcial.html', detecoes=exibir_dados, total=total_real, data_selecionada=alvo)


@app.route('/video_feed/<int:cam_id>', methods=['GET'])
def video_feed(cam_id):
    if cam_id not in camera_state:
        return "Câmara não encontrada", 404

    estado = camera_state[cam_id]

    def generate():
        while True:
            with estado["lock"]:
                if estado["frame"] is None:
                    continue
                _, buffer = cv2.imencode('.jpg', estado["frame"])
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/encerrarApp', methods=['POST'])
def encerraApp():
    os._exit(0)
    print("App foi encerrada!")
    return 'App foi encerrada', 204


def open_browser():
    webbrowser.open_new("http://127.0.0.1:5002")


if __name__ == '__main__':
    Timer(2, open_browser).start()
    app.run(host='127.0.0.1', port=5002, debug=False, use_reloader=False)