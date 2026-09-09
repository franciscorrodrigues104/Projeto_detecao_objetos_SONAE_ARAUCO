import cv2
import threading
import time
import os
import csv
import json
import webbrowser
import logging
import logging.handlers
from datetime import datetime
from collections import defaultdict, deque
from pathlib import Path
from threading import Timer
from queue import Queue, Empty, Full
import sys

import numpy as np
from flask import Flask, Response, render_template, request, jsonify
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

def resource_path(relative_path):
    """
    Compatível com execução normal e PyInstaller.
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

load_dotenv(resource_path(".env"))

# ==================================================
# LOGGING
# ==================================================
PASTA_LOGS = Path("logs")
PASTA_LOGS.mkdir(exist_ok=True)
FICHEIRO_LOG = PASTA_LOGS / "app.log"

logger = logging.getLogger("manta")
logger.setLevel(logging.DEBUG)

_formato = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(threadName)-15s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_handler_ficheiro = logging.handlers.RotatingFileHandler(
    FICHEIRO_LOG, maxBytes=2_000_000, backupCount=5, encoding="utf-8"
)
_handler_ficheiro.setLevel(logging.DEBUG)
_handler_ficheiro.setFormatter(_formato)
logger.addHandler(_handler_ficheiro)
logger.propagate = False

logger_werkzeug = logging.getLogger("werkzeug")
logger_werkzeug.handlers = [_handler_ficheiro]
logger_werkzeug.setLevel(logging.INFO)
logger_werkzeug.propagate = False

try:
    import flask.cli
    flask.cli.show_server_banner = lambda *args, **kwargs: None
except Exception:
    pass

def _log_excecao_thread(args):
    logger.error(
        "Exceção não tratada na thread '%s': %s",
        args.thread.name if args.thread else "?",
        args.exc_value,
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )

threading.excepthook = _log_excecao_thread

logger.info("=" * 60)
logger.info("Aplicação a arrancar")

# ==================================================
# CONFIGURAÇÃO
# ==================================================

MODELO_MANTA = resource_path(
    os.getenv("MODELO_MANTA", "modelo_seg_manta_v2.pt")
)

MODELO_DEFEITOS = resource_path(
    os.getenv("MODELO_DEFEITOS", "modelo_detecao_defeitos_manta_v2.pt")
)

CAMERA_URL = os.getenv("CAMERA_URL")

CONF_MANTA = float(os.getenv("CONF_MANTA", "0.6"))

# Valores por omissão, usados se a variável não estiver definida no .env
# ou se o JSON lá dentro estiver inválido.
_CONFIANCA_POR_CLASSE_DEFAULT = {
    "bordo_NOK": 0.95,
    "monte": 0.4,
}


def carregar_confianca_por_classe():
    """Lê CONFIANCA_POR_CLASSE do .env como um objeto JSON, ex.:

        CONFIANCA_POR_CLASSE={"bordo_NOK": 0.95, "monte": 0.4}

    Nota: dentro do .env, o valor tem aspas duplas por ser JSON — se o teu
    parser de .env tiver problemas com isso, envolve o valor inteiro em
    aspas simples:

        CONFIANCA_POR_CLASSE='{"bordo_NOK": 0.95, "monte": 0.4}'

    Se a variável não existir, ou o JSON for inválido, usa-se
    _CONFIANCA_POR_CLASSE_DEFAULT e fica um aviso no log.
    """
    raw = os.getenv("CONFIANCA_POR_CLASSE")
    if not raw:
        logger.info("CONFIANCA_POR_CLASSE não definido no .env — a usar valores por omissão: %s",
                    _CONFIANCA_POR_CLASSE_DEFAULT)
        return dict(_CONFIANCA_POR_CLASSE_DEFAULT)
    try:
        valor = json.loads(raw)
        if not isinstance(valor, dict) or not valor:
            raise ValueError("CONFIANCA_POR_CLASSE tem de ser um objeto JSON não vazio, ex.: {\"furo\": 0.5}")
        valor = {str(k): float(v) for k, v in valor.items()}
        logger.info("CONFIANCA_POR_CLASSE carregado do .env: %s", valor)
        return valor
    except Exception:
        logger.exception(
            "CONFIANCA_POR_CLASSE inválido no .env (valor lido: %r) — a usar valores por omissão: %s",
            raw, _CONFIANCA_POR_CLASSE_DEFAULT,
        )
        return dict(_CONFIANCA_POR_CLASSE_DEFAULT)


CONFIANCA_POR_CLASSE = carregar_confianca_por_classe()

CONF_DEFEITOS_MIN = min(CONFIANCA_POR_CLASSE.values())


def gerar_tracker_config(caminho="tracker_manta.yaml"):
    novo_track_thresh = round(max(0.05, CONF_DEFEITOS_MIN - 0.1), 3)
    track_high_thresh = round(max(0.05, novo_track_thresh - 0.1), 3)

    conteudo = (
        "# Gerado automaticamente por app.py a partir de CONFIANCA_POR_CLASSE — não editar à mão.\n"
        "tracker_type: bytetrack\n"
        f"track_high_thresh: {track_high_thresh}\n"
        "track_low_thresh: 0.1\n"
        f"new_track_thresh: {novo_track_thresh}\n"
        "track_buffer: 30\n"
        "match_thresh: 0.8\n"
        "fuse_score: true\n"
    )

    Path(caminho).write_text(conteudo, encoding="utf-8")
    logger.info("tracker_manta.yaml gerado (new_track_thresh=%s, track_high_thresh=%s)",
                novo_track_thresh, track_high_thresh)
    return caminho


TRACKER_CONFIG = resource_path("tracker_manta.yaml")
gerar_tracker_config(TRACKER_CONFIG)

DEBUG_DETECOES = os.getenv("DEBUG_DETECOES", "false").lower() == "true"

# Número de "zonas" em que a largura da manta é dividida para o heatmap.
# Guardamos x_normalizado (posição relativa 0-1) em vez de depender só da
# zona já calculada, por isso este valor pode ser alterado no futuro (ex.:
# para 20 ou 30) sem perder a possibilidade de recalcular o heatmap
# histórico com a nova resolução — ver `zona_a_partir_de_x` mais abaixo.
N_ZONAS = 6

# A cada quantos frames repetimos a segmentação completa da manta (modelo
# mais pesado). Nos frames "saltados" reaproveitamos a última máscara/ROI
# conhecida. Valor mais baixo = mais seguro (deteta mais depressa vibração,
# zoom, ou deslocamento da câmara/manta) mas com menor ganho de performance.
# Começa-se conservador em 3 e sobe-se conforme o FPS observado permitir
# (ver /estado_sistema e o overlay no vídeo).
N_FRAMES_MANTA_SKIP = int(os.getenv("N_FRAMES_MANTA_SKIP", "3"))

# Ao fim de quanto tempo (segundos) uma entrada de ids_contados pode ser
# esquecida por inatividade, para o dicionário não crescer indefinidamente
# ao longo de um dia inteiro de produção.
IDS_CONTADOS_TTL_SEGUNDOS = 3600
IDS_CONTADOS_LIMPEZA_INTERVALO = 600  # de quanto em quanto tempo corre a limpeza

# Tamanho máximo da fila de gravação em disco. Se o disco ficar lento (ou
# houver um pico de defeitos), a fila deixa de crescer sem limite — a partir
# daqui, novas deteções são descartadas (com aviso no log) em vez de
# consumirem RAM indefinidamente.
FILA_GRAVACAO_MAXSIZE = 1000

# De quanto em quanto tempo (segundos) o monitor regista fps/latência/fila.
MONITOR_INTERVALO = 30

# ==================================================
# ARMAZENAMENTO LOCAL (CSV por dia + imagens)
# ==================================================
PASTA_DADOS = Path("dados")
PASTA_IMAGENS = PASTA_DADOS / "imagens"

PASTA_DADOS.mkdir(exist_ok=True)
PASTA_IMAGENS.mkdir(exist_ok=True)

CABECALHO_CSV = ["timestamp", "data", "hora", "track_id", "classe", "confianca",
                 "zona", "x_normalizado", "imagem"]


def caminho_csv_do_dia(data_str):
    return PASTA_DADOS / f"detecoes_{data_str}.csv"


def garantir_csv_do_dia(data_str):
    caminho = caminho_csv_do_dia(data_str)
    if not caminho.exists():
        with open(caminho, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CABECALHO_CSV)
    return caminho


def data_hoje():
    return datetime.now().strftime("%Y-%m-%d")


garantir_csv_do_dia(data_hoje())


def zona_a_partir_de_x(x_normalizado, n_zonas=None):
    """Calcula a zona (índice inteiro) a partir da posição normalizada,
    usando SEMPRE o N_ZONAS atual — é isto que permite mudar a resolução
    do heatmap no futuro sem perder informação dos dados já gravados."""
    n_zonas = n_zonas or N_ZONAS
    try:
        x = float(x_normalizado)
    except (TypeError, ValueError):
        return None
    zona = int(x * n_zonas)
    return max(0, min(n_zonas - 1, zona))


# ==================================================
# MODELOS
# ==================================================

logger.info("MODELO_MANTA: %s", MODELO_MANTA)
logger.info("MODELO_DEFEITOS: %s", MODELO_DEFEITOS)
logger.info("CAMERA_URL: %s", CAMERA_URL)

if not os.path.exists(MODELO_MANTA):
    logger.error("Modelo não encontrado: %s", MODELO_MANTA)
    raise FileNotFoundError(f"Modelo não encontrado: {MODELO_MANTA}")

if not os.path.exists(MODELO_DEFEITOS):
    logger.error("Modelo não encontrado: %s", MODELO_DEFEITOS)
    raise FileNotFoundError(f"Modelo não encontrado: {MODELO_DEFEITOS}")

if not CAMERA_URL:
    logger.warning("CAMERA_URL está vazio ou não definido no .env — a captura de vídeo vai falhar.")

try:
    import torch
    logger.info("torch: CUDA disponível = %s (dispositivo = %s)",
                torch.cuda.is_available(),
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
except Exception:
    logger.warning("Não foi possível verificar disponibilidade de CUDA (torch não acessível diretamente).")

try:
    logger.info("A carregar modelo_manta...")
    modelo_manta = YOLO(MODELO_MANTA)
    logger.info("modelo_manta carregado.")

    logger.info("A carregar modelo_defeitos...")
    modelo_defeitos = YOLO(MODELO_DEFEITOS)
    logger.info("modelo_defeitos carregado.")
except Exception:
    logger.exception("Falha ao carregar um dos modelos YOLO")
    raise


# ==================================================
# ESTADO PARTILHADO
# ==================================================

app = Flask(
    __name__,
    template_folder=resource_path("templates"),
    static_folder=resource_path("static")
)

# Sinaliza a todas as threads para terminarem de forma ordenada (usado por
# /encerrarApp antes do os._exit final — ver mais abaixo).
shutdown_event = threading.Event()

latest_raw_frame = None
raw_frame_lock = threading.Lock()
raw_frame_novo = threading.Event()

latest_frame = None
frame_lock = threading.Lock()

# ids_contados: {(track_id, classe): timestamp_ultima_vez_visto}
ids_contados = {}
ids_lock = threading.Lock()

lock_dados = threading.Lock()

contagem_classes = defaultdict(int)
contagem_zonas = [0] * N_ZONAS
ultimas_deteções = deque(maxlen=50)

# --- contadores para FPS/latência real (ver monitor_worker) ---
contador_frames_camera = 0
contador_frames_yolo = 0
soma_latencia_frame = 0.0
contagem_perf_lock = threading.Lock()

# --- último snapshot calculado pelo monitor, consultável pela rota
# /estado_sistema e desenhado como overlay no próprio vídeo ---
estado_sistema = {
    "fps_camera": 0.0,
    "fps_yolo": 0.0,
    "fila_escrita": 0,
    "latencia_media_ms": 0.0,
    "unicos_contados": 0,
}
estado_sistema_lock = threading.Lock()


def limpar_ids_contados_antigos():
    """Remove de ids_contados as entradas com mais de IDS_CONTADOS_TTL_SEGUNDOS
    sem atividade. Chamado periodicamente pelo yolo_worker."""
    limite = time.time() - IDS_CONTADOS_TTL_SEGUNDOS
    with ids_lock:
        antes = len(ids_contados)
        chaves_antigas = [k for k, v in ids_contados.items() if v < limite]
        for k in chaves_antigas:
            del ids_contados[k]
        depois = len(ids_contados)
    if chaves_antigas:
        logger.info("limpeza ids_contados: %d -> %d entradas (removidas %d com mais de %ds de inatividade)",
                    antes, depois, len(chaves_antigas), IDS_CONTADOS_TTL_SEGUNDOS)


def carregar_estatisticas_do_dia():
    """Ao arrancar a aplicação, relê o CSV de hoje e recalcula os contadores,
    para o dashboard não começar sempre a zeros após um restart."""
    caminho = caminho_csv_do_dia(data_hoje())
    if not caminho.exists():
        return
    with open(caminho, newline="", encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            contagem_classes[linha["classe"]] += 1
            zona = zona_a_partir_de_x(linha.get("x_normalizado"))
            if zona is None:
                try:
                    zona = int(linha["zona"])
                except (ValueError, KeyError):
                    zona = None
            if zona is not None and 0 <= zona < N_ZONAS:
                contagem_zonas[zona] += 1
    logger.info("Estatísticas do dia recarregadas do CSV (%d classes).", len(contagem_classes))


carregar_estatisticas_do_dia()


def obter_estatisticas(data_alvo=None):
    """Devolve (histograma_por_classe, contagem_por_zona, total) para uma data."""
    hoje = data_hoje()

    if data_alvo is None or data_alvo == hoje:
        with lock_dados:
            return dict(contagem_classes), list(contagem_zonas), sum(contagem_classes.values())

    histograma = defaultdict(int)
    zonas = [0] * N_ZONAS
    caminho = caminho_csv_do_dia(data_alvo)
    if caminho.exists():
        with open(caminho, newline="", encoding="utf-8") as f:
            for linha in csv.DictReader(f):
                histograma[linha["classe"]] += 1
                zona = zona_a_partir_de_x(linha.get("x_normalizado"))
                if zona is None:
                    try:
                        zona = int(linha["zona"])
                    except (ValueError, KeyError):
                        zona = None
                if zona is not None and 0 <= zona < N_ZONAS:
                    zonas[zona] += 1
    return dict(histograma), zonas, sum(histograma.values())


def obter_distribuicao_horaria(data_alvo=None):
    """Devolve {"08": 12, "09": 21, ...} com o nº de defeitos por hora do dia."""
    data_alvo = data_alvo or data_hoje()
    por_hora = defaultdict(int)
    caminho = caminho_csv_do_dia(data_alvo)
    if caminho.exists():
        with open(caminho, newline="", encoding="utf-8") as f:
            for linha in csv.DictReader(f):
                hora = linha.get("hora", "")
                if len(hora) >= 2:
                    por_hora[hora[:2]] += 1
    return dict(sorted(por_hora.items()))


# ==================================================
# GRAVAÇÃO EM DISCO NUMA THREAD DEDICADA
# ==================================================
# A fila tem um tamanho máximo (FILA_GRAVACAO_MAXSIZE): se o disco ficar
# lento ou houver um pico de defeitos, deixamos de aceitar novos pedidos em
# vez de deixar a fila (e a RAM) crescer sem limite.
fila_gravacao = Queue(maxsize=FILA_GRAVACAO_MAXSIZE)


def guardar_deteccao(track_id, classe, confianca, x_normalizado, imagem_recorte):
    """Chamado pela thread do YOLO: NÃO toca em disco, só enfileira o pedido.
    Se a fila estiver cheia (disco a não conseguir acompanhar), a deteção é
    descartada com um aviso no log em vez de bloquear o YOLO."""
    item = {
        "quando": datetime.now(),
        "track_id": int(track_id),
        "classe": classe,
        "confianca": float(confianca),
        "x_normalizado": float(x_normalizado),
        "imagem_recorte": imagem_recorte,
    }
    try:
        fila_gravacao.put_nowait(item)
    except Full:
        logger.warning(
            "fila_gravacao cheia (%d itens) — deteção descartada (track_id=%s classe=%s). "
            "Disco pode estar lento ou sobrecarregado.",
            FILA_GRAVACAO_MAXSIZE, track_id, classe,
        )


def _processar_gravacao(item):
    """Corre exclusivamente na thread worker_escrita: aqui sim tocamos em disco."""
    agora = item["quando"]
    data_str = agora.strftime("%Y-%m-%d")
    hora_str = agora.strftime("%H:%M:%S")
    track_id = item["track_id"]
    classe = item["classe"]
    confianca = item["confianca"]
    x_normalizado = item["x_normalizado"]
    imagem_recorte = item["imagem_recorte"]

    pasta_dia = PASTA_IMAGENS / data_str
    pasta_dia.mkdir(exist_ok=True)

    classe_segura = "".join(c for c in str(classe) if c.isalnum() or c in ("-", "_")) or "defeito"
    nome_imagem = f"{agora.strftime('%H%M%S')}_{track_id}_{classe_segura}.jpg"
    caminho_imagem = pasta_dia / nome_imagem

    try:
        if imagem_recorte is not None and imagem_recorte.size > 0:
            cv2.imwrite(str(caminho_imagem), imagem_recorte, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        else:
            caminho_imagem = ""
    except Exception:
        logger.exception("Erro ao guardar imagem em %s", caminho_imagem)
        caminho_imagem = ""

    zona = zona_a_partir_de_x(x_normalizado)

    linha = [
        agora.isoformat(),
        data_str,
        hora_str,
        track_id,
        classe,
        round(confianca, 3),
        zona if zona is not None else "",
        round(x_normalizado, 4),
        str(caminho_imagem),
    ]

    caminho_csv = garantir_csv_do_dia(data_str)

    ultimo_erro = None
    for tentativa in range(5):
        try:
            with open(caminho_csv, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(linha)
            ultimo_erro = None
            break
        except PermissionError as e:
            ultimo_erro = e
            logger.warning("CSV bloqueado (tentativa %d/5): %s", tentativa + 1, e)
            time.sleep(0.3)
        except Exception:
            logger.exception("Erro ao gravar no CSV")
            ultimo_erro = None
            break

    if ultimo_erro is not None:
        logger.error(
            "Erro ao gravar no CSV (ficheiro bloqueado após 5 tentativas): %s -> Verifica se '%s' "
            "está aberto no Excel ou noutro programa, e se a pasta '%s' tem permissões de escrita.",
            ultimo_erro, caminho_csv, PASTA_DADOS,
        )

    with lock_dados:
        contagem_classes[classe] += 1
        if zona is not None:
            contagem_zonas[zona] += 1
        ultimas_deteções.appendleft({
            "hora": hora_str,
            "classe": classe,
            "track_id": track_id,
            "confianca": round(confianca, 2),
        })

    logger.info("Deteção gravada: track_id=%s classe=%s conf=%.3f zona=%s x_norm=%.3f imagem=%s",
                track_id, classe, confianca, zona, x_normalizado, caminho_imagem)


def worker_escrita():
    logger.info("worker_escrita: iniciado, à espera de deteções para gravar...")
    while not shutdown_event.is_set():
        try:
            item = fila_gravacao.get(timeout=1)
        except Empty:
            continue
        try:
            _processar_gravacao(item)
        except Exception:
            logger.exception("worker_escrita: erro inesperado a gravar deteção")
        finally:
            fila_gravacao.task_done()
    logger.info("worker_escrita: a terminar (shutdown_event ativo).")


# ==================================================
# WORKER 1: captura de vídeo (thread dedicada e "burra")
# ==================================================

def captura_worker():
    global latest_raw_frame, contador_frames_camera

    logger.info("captura_worker: a abrir stream em %s", CAMERA_URL)
    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        logger.error("captura_worker: não foi possível abrir o stream MJPG em %s", CAMERA_URL)
        raise Exception("Erro ao abrir stream MJPG")
    logger.info("captura_worker: stream aberto com sucesso.")

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        logger.debug("captura_worker: CAP_PROP_BUFFERSIZE não suportado por este backend.")

    falhas_seguidas = 0

    while not shutdown_event.is_set():
        # Alguns backends de stream MJPG "morrem" silenciosamente e passam a
        # reportar cap.isOpened() == False ANTES de cap.read() começar a
        # devolver ret == False. Verificar isto aqui, antes de ler, apanha
        # esses casos mais cedo do que só reagir a leituras falhadas.
        if not cap.isOpened():
            logger.warning("captura_worker: stream não está aberto, a tentar reabrir em 1s...")
            time.sleep(1)
            cap = cv2.VideoCapture(CAMERA_URL)
            if cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                logger.info("captura_worker: stream reaberto com sucesso.")
            else:
                logger.error("captura_worker: reabertura do stream falhou.")
            continue

        ret, frame = cap.read()
        if not ret:
            falhas_seguidas += 1
            logger.warning("captura_worker: falha ao ler frame (falha nº %d seguida). A reabrir stream em 1s...",
                            falhas_seguidas)
            time.sleep(1)
            cap = cv2.VideoCapture(CAMERA_URL)
            if not cap.isOpened():
                logger.error("captura_worker: reabertura do stream falhou.")
            else:
                logger.info("captura_worker: stream reaberto com sucesso.")
            continue

        falhas_seguidas = 0

        with raw_frame_lock:
            latest_raw_frame = frame
        raw_frame_novo.set()

        with contagem_perf_lock:
            contador_frames_camera += 1

    logger.info("captura_worker: a terminar (shutdown_event ativo).")


# ==================================================
# WORKER 3: monitor de FPS / latência / fila de gravação
# ==================================================
# Antes só tínhamos "quantidade de frames processados" espalhado por dois
# sítios diferentes (captura e yolo). Isto centraliza num único lugar,
# regista no log a cada MONITOR_INTERVALO segundos, e disponibiliza o
# resultado em `estado_sistema` — usado tanto pela rota /estado_sistema
# como pelo overlay desenhado no próprio vídeo.

def monitor_worker():
    global contador_frames_camera, contador_frames_yolo, soma_latencia_frame

    logger.info("monitor_worker: iniciado (intervalo=%ds).", MONITOR_INTERVALO)
    while not shutdown_event.is_set():
        if shutdown_event.wait(timeout=MONITOR_INTERVALO):
            break

        with contagem_perf_lock:
            frames_cam = contador_frames_camera
            frames_yolo = contador_frames_yolo
            soma_lat = soma_latencia_frame
            contador_frames_camera = 0
            contador_frames_yolo = 0
            soma_latencia_frame = 0.0

        fps_camera = frames_cam / MONITOR_INTERVALO
        fps_yolo = frames_yolo / MONITOR_INTERVALO
        latencia_media_ms = (soma_lat / frames_yolo * 1000) if frames_yolo else 0.0
        tamanho_fila = fila_gravacao.qsize()
        with ids_lock:
            n_unicos = len(ids_contados)

        with estado_sistema_lock:
            estado_sistema.update(
                fps_camera=round(fps_camera, 1),
                fps_yolo=round(fps_yolo, 1),
                fila_escrita=tamanho_fila,
                latencia_media_ms=round(latencia_media_ms, 1),
                unicos_contados=n_unicos,
            )

        logger.info(
            "monitor: fps_camera=%.1f fps_yolo=%.1f fila_escrita=%d latencia_media=%.1fms unicos=%d",
            fps_camera, fps_yolo, tamanho_fila, latencia_media_ms, n_unicos,
        )
        if tamanho_fila >= FILA_GRAVACAO_MAXSIZE * 0.8:
            logger.warning("monitor: fila_gravacao a aproximar-se do limite (%d/%d) — disco pode estar lento.",
                            tamanho_fila, FILA_GRAVACAO_MAXSIZE)

    logger.info("monitor_worker: a terminar (shutdown_event ativo).")


# ==================================================
# WORKER 2: segmentação da manta + deteção/tracking de defeitos
# ==================================================

def yolo_worker():
    global latest_frame, contador_frames_yolo, soma_latencia_frame

    dia_atual = data_hoje()
    frames_processados = 0
    ultima_limpeza_ids = time.time()
    logger.info("yolo_worker: iniciado, à espera de frames...")

    # cache da última segmentação da manta válida, reaproveitada nos frames
    # "saltados" (ver N_FRAMES_MANTA_SKIP) para poupar uma inferência inteira
    contador_frames_manta = 0
    cache_manta_valida = False
    cache_mask = None
    cache_bbox = None  # (x, y, w, h)

    while not shutdown_event.is_set():
        recebeu = raw_frame_novo.wait(timeout=1.0)
        if not recebeu:
            logger.debug("yolo_worker: sem frames novos no último 1s (a captura pode estar parada).")

        with raw_frame_lock:
            frame = latest_raw_frame
            raw_frame_novo.clear()

        if frame is None:
            continue

        _inicio_frame = time.time()

        try:
            frame = frame.copy()

            # --- reset diário dos contadores em memória (o CSV mantém sempre o histórico completo) ---
            novo_dia = data_hoje()
            if novo_dia != dia_atual:
                logger.info("yolo_worker: novo dia detetado (%s -> %s), a repor contadores.", dia_atual, novo_dia)
                dia_atual = novo_dia
                garantir_csv_do_dia(dia_atual)
                with lock_dados:
                    contagem_classes.clear()
                    for i in range(N_ZONAS):
                        contagem_zonas[i] = 0
                    ultimas_deteções.clear()
                with ids_lock:
                    ids_contados.clear()

            # --- limpeza periódica de ids_contados (evita crescimento indefinido) ---
            agora_ts = time.time()
            if agora_ts - ultima_limpeza_ids >= IDS_CONTADOS_LIMPEZA_INTERVALO:
                limpar_ids_contados_antigos()
                ultima_limpeza_ids = agora_ts

            # =============================================
            # SEGMENTAÇÃO DA MANTA (com skip de frames)
            # =============================================

            usar_deteccao_manta = (contador_frames_manta % N_FRAMES_MANTA_SKIP == 0) or not cache_manta_valida
            contador_frames_manta += 1

            painel_roi = np.zeros_like(frame)

            if usar_deteccao_manta:
                _t0 = time.time()
                resultado_manta = modelo_manta(frame, conf=CONF_MANTA, verbose=False)[0]
                _dt_manta = time.time() - _t0
                if frames_processados == 0:
                    logger.info("yolo_worker: 1ª inferência do modelo_manta demorou %.2fs (inclui warmup).", _dt_manta)
                elif _dt_manta > 1.0:
                    logger.warning("yolo_worker: inferência do modelo_manta lenta (%.2fs).", _dt_manta)
                frame_segmentado = resultado_manta.plot()

                if resultado_manta.masks is None:
                    logger.debug("yolo_worker: nenhuma manta segmentada neste frame.")
                    cache_manta_valida = False
                    cache_mask = None
                    cache_bbox = None
                else:
                    maior_area = 0
                    melhor_polygon = None
                    for polygon in resultado_manta.masks.xy:
                        polygon = polygon.astype(np.int32)
                        area = cv2.contourArea(polygon)
                        if area > maior_area:
                            maior_area = area
                            melhor_polygon = polygon

                    if melhor_polygon is not None:
                        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(mask, [melhor_polygon], 255)
                        kernel = np.ones((2, 2), np.uint8)
                        mask = cv2.erode(mask, kernel, iterations=1)

                        x, y, w, h = cv2.boundingRect(melhor_polygon)

                        cache_mask = mask
                        cache_bbox = (x, y, w, h)
                        cache_manta_valida = True
                    else:
                        cache_manta_valida = False
                        cache_mask = None
                        cache_bbox = None
            else:
                # frame "saltado": reaproveita a última máscara/bbox conhecida,
                # aplicada sobre o frame ATUAL
                frame_segmentado = frame.copy()
                if cache_bbox is not None:
                    x, y, w, h = cache_bbox
                    cv2.rectangle(frame_segmentado, (x, y), (x + w, y + h), (0, 200, 255), 2)
                    cv2.putText(frame_segmentado, "manta (cache)", (x, max(y - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            if cache_manta_valida and cache_mask is not None and cache_bbox is not None:
                x, y, w, h = cache_bbox
                frame_mascarado = cv2.bitwise_and(frame, frame, mask=cache_mask)
                roi = frame_mascarado[y:y + h, x:x + w]

                if roi.size > 0 and w > 0:

                    # =====================================
                    # TRACKING DE DEFEITOS (corre em TODOS os frames)
                    # =====================================

                    _t0 = time.time()
                    resultado_defeitos = modelo_defeitos.track(
                        roi, conf=CONF_DEFEITOS_MIN, persist=True, verbose=False,
                        tracker=TRACKER_CONFIG
                    )[0]
                    _dt_defeitos = time.time() - _t0
                    if frames_processados == 0:
                        logger.info("yolo_worker: 1ª inferência do modelo_defeitos demorou %.2fs (inclui warmup).", _dt_defeitos)
                    elif _dt_defeitos > 1.0:
                        logger.warning("yolo_worker: inferência do modelo_defeitos lenta (%.2fs).", _dt_defeitos)

                    nomes = modelo_defeitos.names

                    def nome_da_classe(cls_idx):
                        if isinstance(nomes, dict):
                            return nomes.get(int(cls_idx), str(cls_idx))
                        return nomes[int(cls_idx)] if int(cls_idx) < len(nomes) else str(cls_idx)

                    if DEBUG_DETECOES and resultado_defeitos.boxes is not None and len(resultado_defeitos.boxes) > 0:
                        ids_dbg = resultado_defeitos.boxes.id
                        cls_arr = resultado_defeitos.boxes.cls.cpu().numpy()
                        conf_arr = resultado_defeitos.boxes.conf.cpu().numpy()
                        for i in range(len(cls_arr)):
                            tid_dbg = int(ids_dbg[i]) if ids_dbg is not None else None
                            logger.debug("classe=%s conf=%.3f track_id=%s",
                                         nome_da_classe(cls_arr[i]), conf_arr[i], tid_dbg)

                    if resultado_defeitos.boxes is not None and len(resultado_defeitos.boxes) > 0:
                        classes_idx_todas = resultado_defeitos.boxes.cls.cpu().numpy().astype(int)
                        confs_todas = resultado_defeitos.boxes.conf.cpu().numpy()

                        manter = np.array([
                            conf >= CONFIANCA_POR_CLASSE.get(nome_da_classe(cls_idx), CONF_DEFEITOS_MIN)
                            for cls_idx, conf in zip(classes_idx_todas, confs_todas)
                        ])

                        resultado_defeitos = resultado_defeitos[manter]

                    roi_anotada = resultado_defeitos.plot()

                    if (resultado_defeitos.boxes is not None and
                            resultado_defeitos.boxes.id is not None):

                        ids = resultado_defeitos.boxes.id.cpu().numpy().astype(int)
                        classes_idx = resultado_defeitos.boxes.cls.cpu().numpy().astype(int)
                        confs = resultado_defeitos.boxes.conf.cpu().numpy()
                        xyxy = resultado_defeitos.boxes.xyxy.cpu().numpy()

                        for track_id, cls_idx, conf, box in zip(ids, classes_idx, confs, xyxy):

                            classe_nome = nome_da_classe(cls_idx)

                            chave = (int(track_id), classe_nome)
                            agora_ts = time.time()
                            with ids_lock:
                                if chave in ids_contados:
                                    ids_contados[chave] = agora_ts
                                    continue
                                ids_contados[chave] = agora_ts

                            bx1, by1, bx2, by2 = box.astype(int)
                            # Limitar SEMPRE os dois extremos (min E max) às dimensões
                            # da ROI: antes só se corrigia bx1/by1 (>= 0), mas bx2/by2
                            # também podem exceder roi.shape se a caixa do tracker sair
                            # ligeiramente da imagem, dando um recorte vazio/errado.
                            bx1 = max(bx1, 0)
                            by1 = max(by1, 0)
                            bx2 = min(bx2, roi.shape[1])
                            by2 = min(by2, roi.shape[0])
                            recorte = roi[by1:by2, bx1:bx2]

                            centro_x = (bx1 + bx2) / 2
                            x_normalizado = max(0.0, min(1.0, centro_x / w))

                            guardar_deteccao(track_id, classe_nome, conf, x_normalizado, recorte)

                    with ids_lock:
                        n_unicos = len(ids_contados)
                    cv2.putText(roi_anotada, f"Visiveis: {len(resultado_defeitos.boxes)}",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(roi_anotada, f"Unicos (ultima hora): {n_unicos}",
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    painel_roi = cv2.resize(roi_anotada, (frame.shape[1], frame.shape[0]))
                else:
                    logger.debug("yolo_worker: ROI vazia após máscara (w=%s, roi.size=%s).", w, roi.size)

            # --- overlay com FPS/latência/fila em tempo real (atualizado a cada
            # MONITOR_INTERVALO segundos pelo monitor_worker) — ajuda muito a
            # confirmar em produção que tudo está fluido, sem ter de ir ao log.
            with estado_sistema_lock:
                _fps_cam = estado_sistema["fps_camera"]
                _fps_yolo = estado_sistema["fps_yolo"]
                _fila = estado_sistema["fila_escrita"]  

            dashboard_frame = np.hstack([frame_segmentado, painel_roi])

            with frame_lock:
                latest_frame = dashboard_frame

            if frames_processados == 0:
                logger.info("yolo_worker: primeiro frame processado com sucesso, latest_frame já disponível.")

            frames_processados += 1

            tempo_processamento = time.time() - _inicio_frame
            with contagem_perf_lock:
                contador_frames_yolo += 1
                soma_latencia_frame += tempo_processamento

        except Exception:
            logger.exception("yolo_worker: erro inesperado a processar frame")

    logger.info("yolo_worker: a terminar (shutdown_event ativo).")


threading.Thread(target=captura_worker, daemon=True, name="captura").start()
threading.Thread(target=yolo_worker, daemon=True, name="yolo").start()
threading.Thread(target=worker_escrita, daemon=True, name="escrita").start()
threading.Thread(target=monitor_worker, daemon=True, name="monitor").start()

# ==================================================
# ROTAS FLASK
# ==================================================

@app.route('/video_feed')
def video_feed():
    cliente = request.remote_addr  # capturado aqui, dentro do contexto do pedido
    logger.info("Pedido recebido em /video_feed (cliente=%s)", cliente)

    def generate():
        frames_enviados = 0
        avisou_sem_frame = False
        try:
            while True:
                with frame_lock:
                    if latest_frame is None:
                        if not avisou_sem_frame:
                            logger.warning("/video_feed: ainda não há nenhum frame processado (latest_frame is None).")
                            avisou_sem_frame = True
                        continue
                    avisou_sem_frame = False
                    _, buffer = cv2.imencode('.jpg', latest_frame)
                frames_enviados += 1
                if frames_enviados == 1:
                    logger.info("/video_feed: a enviar o primeiro frame para o browser.")
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except GeneratorExit:
            logger.info("/video_feed: cliente (%s) fechou a ligação após %d frames enviados.",
                        cliente, frames_enviados)
            raise
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/dados_grafico')
def dados_grafico():
    """Devolve, em JSON, os dados para o histograma (por classe de defeito),
    o heatmap (por zona da manta) e a distribuição por hora, para a data
    pedida (ou hoje)."""
    data_alvo = request.args.get('data') or data_hoje()
    histograma, zonas, total = obter_estatisticas(data_alvo)
    por_hora = obter_distribuicao_horaria(data_alvo)
    top_defeitos = dict(sorted(histograma.items(), key=lambda kv: kv[1], reverse=True))
    return jsonify({
        "histograma": histograma,
        "top_defeitos": top_defeitos,
        "heatmap": zonas,
        "por_hora": por_hora,
        "total": total,
        "data": data_alvo,
    })


@app.route('/estado_sistema')
def estado_sistema_route():
    """Snapshot de FPS câmara/YOLO, latência média e tamanho da fila de
    gravação — pensado para um pequeno painel de monitorização em produção,
    sem teres de ir ao ficheiro de log."""
    with estado_sistema_lock:
        return jsonify(dict(estado_sistema))


@app.route('/')
def index():
    logger.info("Pedido recebido em / (cliente=%s)", request.remote_addr)
    data_filtro = request.args.get('data')
    data_alvo = data_filtro or data_hoje()
    histograma, zonas, total = obter_estatisticas(data_alvo)

    return render_template(
        'index.html',
        data_selecionada=data_alvo,
        total=total,
        n_zonas=N_ZONAS,
        histograma_inicial=json.dumps(histograma),
        heatmap_inicial=json.dumps(zonas),
    )


@app.route('/encerrarApp', methods=['POST'])
def encerrar_app():
    """Encerramento (um pouco) mais gracioso: sinaliza as threads para
    pararem e dá até 5s para a fila de gravação esvaziar (imagens/CSV
    pendentes) antes de matar o processo. Para uma aplicação local isto não
    é crítico, mas evita perder deteções que já estavam na fila."""
    logger.info("Pedido de encerramento recebido em /encerrarApp")
    shutdown_event.set()

    pendentes = fila_gravacao.qsize()
    if pendentes:
        logger.info("A aguardar até 5s por %d gravação(ões) pendente(s) na fila...", pendentes)
        inicio = time.time()
        while not fila_gravacao.empty() and time.time() - inicio < 5:
            time.sleep(0.1)
        if not fila_gravacao.empty():
            logger.warning("Encerramento: ainda ficaram %d gravação(ões) por processar.",
                            fila_gravacao.qsize())

    logger.info("A encerrar a aplicação.")
    os._exit(0)


def open_browser():
    logger.info("A abrir browser em http://127.0.0.1:5004")
    webbrowser.open_new("http://127.0.0.1:5004")


if __name__ == '__main__':
    Timer(2, open_browser).start()
    logger.info("Servidor Flask a arrancar em http://127.0.0.1:5004")
    app.run(host='127.0.0.1', port=5004, debug=False, use_reloader=False)