import cv2
from flask import Flask, Response, render_template, request
import time
from datetime import datetime
import os
import webbrowser
from threading import Timer
import sys
from supabase import create_client, Client
import threading
from dotenv import load_dotenv
import logging
import getpass
import winsound
from win11toast import toast



load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Nome da tabela e do bucket vêm do .env — têm de ser os MESMOS valores
# usados na app de deteção, para o viewer mostrar os dados certos.
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
log_file = open("logs.txt", "a", encoding="utf-8", buffering=1)
sys.stdout = log_file
sys.stderr = log_file

alarme_ativo = False
alarme_lock = threading.Lock()

def loop_alarme():
    global alarme_ativo

    print("[ALARME] Thread iniciada.")

    while True:

        with alarme_lock:
            if not alarme_ativo:
                print("[ALARME] Thread terminada.")
                break

        tocar_buzzer()

        time.sleep(13)

def iniciar_alarme():
    global alarme_ativo

    with alarme_lock:

        # já existe um alarme a tocar
        if alarme_ativo:
            return

        alarme_ativo = True

    threading.Thread(
        target=loop_alarme,
        daemon=True
    ).start()


def parar_alarme():
    global alarme_ativo

    with alarme_lock:
        alarme_ativo = False

    print("[ALARME] Pedido para parar.")

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# Estado global — guarda os IDs já vistos, para detetar o que é novo
ids_conhecidos = set()
ids_lock = threading.Lock()
primeira_chamada = True  # evita apitar por tudo o que já existe quando a app arranca


def enviar_notificacao_windows(quantidade, nome_camara=None):
    def _notificar():
        try:
            texto = f"Foi detetada uma falha ou desalinhamento dos lotes!"
            toast(
                "Nova falha detetada",
                texto,
                icon=resource_path(os.path.join("static", "aviso.png")),
                scenario="reminder",
                button="Fechar",
                audio={"silent": "true"}
            )

        except Exception as e:
            print(f"[NOTIF] Erro ao mostrar notificação: {e}")
    threading.Thread(target=_notificar, daemon=True).start()


def monitorizar_novas_falhas():
    global primeira_chamada

    print("[MONITOR] Monitorização iniciada.")

    while True:

        try:
            hoje_cpu = datetime.now().strftime('%Y-%m-%d')

            response = supabase.table(SUPABASE_TABLE)\
                .select("*")\
                .filter("timestamp_inicio", "ilike", f"{hoje_cpu}%")\
                .execute()

            dados = response.data

            ids_atuais = {linha["id"] for linha in dados}

            # Só a ÚLTIMA deteção de hoje importa para decidir se o alarme
            # deve continuar a tocar. As anteriores já foram tratadas
            # (checkbox fica disabled assim que é marcado), por isso olhar
            # para "existe alguma pendente" faria o alarme nunca parar caso
            # fique alguma deteção antiga por checkar.
            ultima_pendente = False
            if dados:
                ultima_deteccao = max(dados, key=lambda linha: linha["id"])
                ultima_pendente = ultima_deteccao.get("check", 0) == 0

            with ids_lock:

                if primeira_chamada:

                    ids_conhecidos.clear()
                    ids_conhecidos.update(ids_atuais)
                    primeira_chamada = False

                    print(
                        f"[MONITOR] Estado inicial carregado "
                        f"({len(ids_conhecidos)} deteções)."
                    )

                    # Se já arrancarmos com a última deteção por checkar,
                    # o alarme deve tocar desde já.
                    if ultima_pendente:
                        iniciar_alarme()

                else:

                    novos = ids_atuais - ids_conhecidos

                    if novos:

                        print(
                            f"[MONITOR] Novas falhas encontradas: "
                            f"{novos}"
                        )

                        iniciar_alarme()
                        enviar_notificacao_windows(len(novos))

                    ids_conhecidos.update(ids_atuais)

            # Sincroniza o estado do alarme com a BD, em TODAS as
            # instâncias que estejam a correr esta app. É isto que garante
            # que, quando o check é feito numa instância, o alarme para
            # também nas outras (até 5s depois, no pior caso).
            if ultima_pendente:
                iniciar_alarme()
            else:
                parar_alarme()

        except Exception as e:
            print(f"[MONITOR] Erro: {e}")

        time.sleep(5)


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

app = Flask(__name__, template_folder=resource_path("templates"), static_folder=resource_path("static"))

# Estado partilhado por câmara: só o frame para exibição no /video_feed.
camera_state = {
    cam["id"]: {"lock": threading.Lock(), "frame": None}
    for cam in CAMERAS
}


def camera_capture_loop(camera):
    """
    Loop de captura para UMA câmara. Só lê frames e atualiza o estado
    partilhado para exibição — sem qualquer deteção/inferência.
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

        frame = cv2.resize(frame, (800, 450))

        with estado["lock"]:
            estado["frame"] = frame


# Inicia uma thread de captura por câmara.
for cam in CAMERAS:
    threading.Thread(target=camera_capture_loop, args=(cam,), daemon=True).start()



@app.route('/', methods=['GET'])
def index():
    data_filtro = request.args.get('data')
    hoje_cpu = datetime.now().strftime('%Y-%m-%d')

    try:
        query = supabase.table(SUPABASE_TABLE).select("*")

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
    
    
@app.route('/atualizar_check/<int:detecao_id>', methods=['POST'])
def atualizar_check(detecao_id):
    try:
        dados = request.get_json(silent=True) or {}
        novo_valor = 1 if dados.get("check") else 0

        utilizador = getpass.getuser()
        timestamp_check = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Atualiza o estado atual na própria deteção (consulta rápida, sem join).
        supabase.table(SUPABASE_TABLE)\
            .update({
                "check": novo_valor,
                "utilizador_check": utilizador,
                "timestamp_check": timestamp_check,
            })\
            .eq("id", detecao_id)\
            .execute()

        # Regista esta ação como uma nova linha no histórico — nunca apaga
        # nem sobrescreve as anteriores, por isso fica o registo completo.
        supabase.table("historico_checks").insert({
            "detecao_id": detecao_id,
            "user": utilizador,
            "acao": "check" if novo_valor else "uncheck",
            "timestamp": timestamp_check,
        }).execute()

        # Feedback imediato NESTA instância (não espera pelo próximo ciclo
        # do monitor, que corre a cada 5s). As restantes instâncias vão
        # sincronizar o alarme sozinhas, através de monitorizar_novas_falhas,
        # que lê o estado real da última deteção diretamente da BD.
        if novo_valor == 1:
            parar_alarme()

        return {"sucesso": True, "check": novo_valor, "utilizador": utilizador, "timestamp_check": timestamp_check}, 200
    except Exception as e:
        print(f"Erro ao atualizar check (id={detecao_id}): {e}")
        return {"sucesso": False}, 500


@app.route('/historico_checks/<int:detecao_id>', methods=['GET'])
def historico_checks(detecao_id):
    try:
        response = supabase.table("historico_checks")\
            .select("*")\
            .eq("detecao_id", detecao_id)\
            .order("id", desc=True)\
            .execute()
        return {"sucesso": True, "historico": response.data}, 200
    except Exception as e:
        print(f"Erro ao obter histórico (detecao_id={detecao_id}): {e}")
        return {"sucesso": False, "historico": []}, 500


def tocar_buzzer():
    def _play():
        try:
            caminho_som = resource_path(os.path.join("static", "buzzer (3).wav"))
            winsound.PlaySound(caminho_som, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as e:
            print(f"Erro ao tocar som: {e}")
    threading.Thread(target=_play, daemon=True).start()


@app.route('/tabela_atualizada', methods=['GET'])
def tabela_atualizada():
    global primeira_chamada

    data_filtro = request.args.get('data')
    hoje_cpu = datetime.now().strftime('%Y-%m-%d')
    alvo = data_filtro if data_filtro else hoje_cpu

    try:
        response = supabase.table(SUPABASE_TABLE)\
            .select("*")\
            .filter("timestamp_inicio", "ilike", f"{alvo}%")\
            .order("counter_dia", desc=True)\
            .execute()

        dados = response.data
        total_real = len(dados)
        exibir_dados = dados

    except Exception as e:
        print(f"Erro: {e}")
        exibir_dados = []
        total_real = 0

    if alvo != hoje_cpu:
        print(f"[INFO] Consulta histórica ({alvo})")

    return render_template(
        'tabela_parcial.html',
        detecoes=exibir_dados,
        total=total_real,
        data_selecionada=alvo
    )


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
    print("App foi encerrada!")
    
    def desligar():
        time.sleep(0.5)  
        os._exit(0)
    
    threading.Thread(target=desligar).start()
    return 'App foi encerrada', 204


def open_browser():
    webbrowser.open_new("http://127.0.0.1:5003")


if __name__ == '__main__':
    Timer(2, open_browser).start()

    threading.Thread(
        target=monitorizar_novas_falhas,
        daemon=True
    ).start()

    app.run(host='127.0.0.1', port=5003, debug=False, use_reloader=False,threaded=True)