import cv2
import threading
import time
import sys
import os
from datetime import datetime
import io
from flask import Flask, Response, render_template, request
from ultralytics import YOLO
from threading import Timer
import webbrowser
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
bucket_name = "fotos_detecao"
CAMERA_URL = os.getenv("CAMERA_URL")

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

log_file = open("logs.txt", "a", encoding="utf-8")
sys.stdout = log_file
sys.stderr = log_file

model = YOLO("best.pt")
app = Flask(__name__)

latest_frame = None
frame_lock = threading.Lock()

def guarda_bd(t_inicio, t_fim, t_total, imagem):
    try:
        _, buffer = cv2.imencode('.jpg', imagem, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        file_bits = io.BytesIO(buffer).read()

        file_path = f"foto_{t_inicio}.jpg"
        supabase.storage.from_(bucket_name).upload(file_path, file_bits, {"content-type": "image/jpeg"})
        url_publica = supabase.storage.from_(bucket_name).get_public_url(file_path)

        data_hoje = datetime.now().strftime('%Y-%m-%d')

        response_count = supabase.table("detecoes")\
            .select("*", count="exact")\
            .filter("timestamp_inicio", "ilike", f"{data_hoje}%")\
            .execute()
        
        proximo_contador = (response_count.count or 0) + 1

        data = {
            "timestamp_inicio": t_inicio,
            "timestamp_fim": t_fim,
            "duracao_total": t_total,
            "veiculo": "Trator",
            "counter_dia": proximo_contador,
            "imagem_detecao": url_publica
        }
        
        supabase.table("detecoes").insert(data).execute()
        print(f"Inserido na base de dados: {t_total}s (Contador: {proximo_contador})")
    except Exception as e:
        print(f"Erro ao guardar na base de dados: {e}")

def yolo_worker():
    global latest_frame
    cap = cv2.VideoCapture(CAMERA_URL)
    
    trator_na_cena = False
    timestamp_inicio_real = None
    ultimo_momento_visto = 0
    aviso_perda_enviado = False
    ultimo_momento_frame = time.time()
    tempo_acumulado_visto = 0
    frame_para_guardar = None
    ja_capturou_imagem = False
    TEMPO_ESPERA_PARA_FECHAR = 30  
    TEMPO_MINIMO_DETECAO = 6 

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            time.sleep(2)
            cap = cv2.VideoCapture(CAMERA_URL)
            continue
            
        results = model(frame, verbose=False, conf=0.9) 
        annotated_frame = results[0].plot()

       #faz o cálculo do centro do ecrã
        altura, largura, _ = frame.shape
        meio_do_ecra = largura // 2
        
        detetou_no_lado_correto = False

        #encontra objetos e calcula onde está o quadrado do YOLO (com o eixo x)
        for box in results[0].boxes:
            x1, y1, x2, y2= box.xyxy[0].tolist()
            centro_x = (x1 + x2) / 2
            
            #caso esteja antes de metade do ecrã, continua (a classe tem que ser 0 de trator) 
            if int(box.cls[0]) == 0 and centro_x < meio_do_ecra: #sem o "and" para  detetar o ecrã completo
                detetou_no_lado_correto = True
                #detetou_trator = True
                break 

        tempo_atual = time.time()
        dt = tempo_atual - ultimo_momento_frame
        ultimo_momento_frame = tempo_atual 

        #ja sabemos que detetou e do lado correto, vai acumular o tempo que o YOLO vê o objeto
        if detetou_no_lado_correto: #if detetou_trator: - deteta no ecrã todo
            ultimo_momento_visto = tempo_atual
            tempo_acumulado_visto += dt 

        #caso o trator desapareça
            if aviso_perda_enviado:
                print("O YOLO voltou a detetar o trator!", flush=True)
                aviso_perda_enviado = False
            
            #deteta o trator
            if not trator_na_cena:
                timestamp_inicio_real = datetime.now()
                trator_na_cena = True
                print(f"Trator detetado: {timestamp_inicio_real.strftime('%H:%M:%S')}")

            #tira print 2 segundos depois do trator aparecer
            if trator_na_cena and not ja_capturou_imagem:
                diff = (datetime.now() - timestamp_inicio_real).total_seconds()
                if diff >= 2.0:
                    frame_para_guardar = annotated_frame.copy()
                    ja_capturou_imagem = True
                    print("Imagem capturada.")


#verifica se o tempo que não deteta o trator é superior ao tempo definifo de espera sem trator (30s)
        if trator_na_cena:
            tempo_sem_ver = tempo_atual - ultimo_momento_visto

            if not detetou_no_lado_correto and not aviso_perda_enviado and tempo_sem_ver < TEMPO_ESPERA_PARA_FECHAR: #tirar o "not detetou_no_lado_correto" para fazer com o ecrã todo
                print(f"Sem trator... a aguardar {TEMPO_ESPERA_PARA_FECHAR}s.", flush=True)
                aviso_perda_enviado = True

#caso o tempo em que o trator nao se ve for superiror ao tempo definido (30s)
            if tempo_sem_ver > TEMPO_ESPERA_PARA_FECHAR:

                t_fim_dt = datetime.fromtimestamp(ultimo_momento_visto)
                tempo_totaldelta = t_fim_dt - timestamp_inicio_real
                tempo_total_segundos = tempo_totaldelta.total_seconds()

#verifica se o trator foi detetado, pelo menos, durante o tempo previamente definido (6s)
                if tempo_acumulado_visto >= TEMPO_MINIMO_DETECAO:
                    t_inicio_str = timestamp_inicio_real.strftime('%Y-%m-%d %H:%M:%S')
                    t_fim_str = t_fim_dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    #guarda na base de dados
                    guarda_bd(t_inicio_str, t_fim_str, round(tempo_total_segundos, 2), frame_para_guardar)
                    print(f"Sessão guardada: {round(tempo_total_segundos, 2)}s (Visto: {round(tempo_acumulado_visto, 2)}s)")
                else:

                    print(f"Não registado. Tempo insuficiente: Visto {round(tempo_acumulado_visto, 2)}s de um total de {round(tempo_total_segundos, 2)}s")
                
                trator_na_cena = False
                timestamp_inicio_real = None
                tempo_acumulado_visto = 0.0
                aviso_perda_enviado = False
                frame_para_guardar = None
                ja_capturou_imagem = False
        
        with frame_lock:
            latest_frame = cv2.resize(annotated_frame, (800, 450))

threading.Thread(target=yolo_worker, daemon=True).start()

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if latest_frame is None: continue
                _, buffer = cv2.imencode('.jpg', latest_frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/', methods= ['GET'])
def index():
    data_filtro = request.args.get('data')
    hoje_cpu = datetime.now().strftime('%Y-%m-%d')
    
    try:
        query = supabase.table("detecoes").select("*")

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
        response = supabase.table("detecoes")\
            .select("*")\
            .filter("timestamp_inicio", "ilike", f"{alvo}%")\
            .order("counter_dia", desc=True).execute()
        
        dados = response.data
        total_real = len(dados)
        exibir_dados = dados
    except Exception as e:
        exibir_dados = [], 0

    return render_template('tabela_parcial.html', detecoes=exibir_dados, total=total_real, data_selecionada=alvo)

@app.route('/encerrarApp', methods=['POST'])
def encerraApp():
    os._exit(0)
    print("App foi encerrada!")
    return 'App foi encerrada', 204

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == '__main__':
    Timer(2, open_browser).start()
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)