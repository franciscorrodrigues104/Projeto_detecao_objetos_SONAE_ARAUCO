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

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
bucket_name = "fotos_detecao_falhas"


log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
log_file = open("logs.txt", "a", encoding="utf-8", buffering=1)
sys.stdout = log_file
sys.stderr = log_file


CAMERA_URL = os.getenv("CAMERA_URL")

frame_lock = threading.Lock()
latest_frame = None


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

modelo_lotes = YOLO(resource_path("modelo_lotes_v2.pt"))
modelo_falhas = YOLO(resource_path("modelo_falhas_lotes_v3.pt"))

app = Flask(__name__, template_folder=resource_path("templates"), static_folder=resource_path("static"))

insercao_lock = threading.Lock()

def guarda_bd(t_inicio, imagem, imagem_original, area):
    with insercao_lock:
        try:
            _, buffer = cv2.imencode('.jpg', imagem, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            file_bits = io.BytesIO(buffer).read()
            
            timestamp = f"foto_{t_inicio}.jpg"

            supabase.storage.from_(bucket_name).upload(
                timestamp, 
                file_bits, 
                {"content-type": "image/jpeg", "upsert": "true"}
            )
            
            _, buff_orig = cv2.imencode('.jpg', imagem_original, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            name_orig = f"original_{t_inicio.replace(' ', '_').replace(':', '-')}.jpg"
            supabase.storage.from_(bucket_name).upload(name_orig, io.BytesIO(buff_orig).read(), {"content-type": "image/jpeg", "upsert": "true"})
            url_orig = supabase.storage.from_(bucket_name).get_public_url(name_orig)
            
            url_publica = supabase.storage.from_(bucket_name).get_public_url(timestamp)

            data_hoje = datetime.now().strftime('%Y-%m-%d')
            response_count = supabase.table("detecoes_falhas")\
                .select("*", count="exact")\
                .filter("timestamp_inicio", "ilike", f"{data_hoje}%")\
                .execute()
            
            proximo_contador = (response_count.count or 0) + 1

            data = {
                "timestamp_inicio": t_inicio,
                "counter_dia": proximo_contador,
                "imagem_detecao": url_publica,
                "imagem_original": url_orig,
                "area_falha": area
            }
            
            supabase.table("detecoes_falhas").insert(data).execute()
            print(f"Inserido com sucesso: (Contador: {proximo_contador})")

        except Exception as e:
            print(f"Erro ao guardar na base de dados: {e}")
            

def gen_frames():
    global latest_frame
    cap = cv2.VideoCapture(CAMERA_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    lote_ja_anunciado = False
    frames_sem_lote = 0  
    TOLERANCIA_FRAMES = 10
    ultima_detecao = datetime.now() - timedelta(seconds=6)

    while True:
        if not cap.isOpened():
            print("Câmara desligada, a tentar reconectar...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(CAMERA_URL)
            continue

        success, frame = cap.read()
        if not success:
            print("Falha ao ler frame, a reiniciar stream...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(CAMERA_URL)
            continue
            
        frame = cv2.resize(frame, (640, 360))
        annotated_frame = frame.copy()

        agora = datetime.now()
        cooldown_ativo = (agora - ultima_detecao).total_seconds() < 6

        results_lotes = modelo_lotes(frame, verbose=False, conf=0.8)
        boxes_lote = [box for box in results_lotes[0].boxes if int(box.cls[0]) == 0]

        falha_detetada_neste_frame = False
        img_para_enviar = None
        area_px = None
        imagem_original = None

        if len(boxes_lote) > 0:
            frames_sem_lote = 0

            if not lote_ja_anunciado:
                print("--- Novo Lote na Linha ---")
                lote_ja_anunciado = True

            if not cooldown_ativo:  
                for box in boxes_lote:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    lote_roi = frame[y1:y2, x1:x2]
                    imagem_original = lote_roi.copy()

                    if lote_roi.size > 0:
                        results_falhas = modelo_falhas(lote_roi, verbose=False, conf=0.60)

                        for box_falha in results_falhas[0].boxes:
                            xf1, yf1, xf2, yf2 = map(int, box_falha.xyxy[0])
                            largura_px = xf2 - xf1
                            altura_px = yf2 - yf1
                            area_px = largura_px * altura_px
                            print(f"area: {area_px}")
                            falha_detetada_neste_frame = True
                            img_para_enviar = results_falhas[0].plot()
                            print(">>> Falha detetada!")

        else:
            frames_sem_lote += 1
            if frames_sem_lote > TOLERANCIA_FRAMES:
                lote_ja_anunciado = False
                

        if falha_detetada_neste_frame:
            ultima_detecao = agora  
            threading.Thread(
                target=guarda_bd,
                args=(agora.strftime('%Y-%m-%d %H:%M:%S'), img_para_enviar, imagem_original, area_px)
            ).start()

        with frame_lock:
            latest_frame = cv2.resize(annotated_frame, (800, 450))
            
threading.Thread(target=gen_frames, daemon=True).start()


@app.route('/', methods= ['GET'])
def index():
    data_filtro = request.args.get('data')
    hoje_cpu = datetime.now().strftime('%Y-%m-%d')
    
    try:
        query = supabase.table("detecoes_falhas").select("*")

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
                           total=total_real)

@app.route('/tabela_atualizada', methods=['GET'])
def tabela_atualizada():
    data_filtro = request.args.get('data')
    hoje_cpu = datetime.now().strftime('%Y-%m-%d')
    alvo = data_filtro if data_filtro else hoje_cpu

    try:
        response = supabase.table("detecoes_falhas")\
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

@app.route('/video_feed', methods=['GET'])
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if latest_frame is None: continue
                _, buffer = cv2.imencode('.jpg', latest_frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/encerrarApp', methods=['POST'])
def encerraApp():
    os._exit(0)
    print("App foi encerrada!")
    return 'App foi encerrada', 204

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5001")

if __name__ == '__main__':
    Timer(2, open_browser).start()
    app.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)