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

# loads data from the .env file
load_dotenv()

# gets supabase data from the .env file and creates supabase client for connection
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
bucket_name = "fotos_detecao"
# gets the camera url
CAMERA_URL = os.getenv("CAMERA_URL")

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# redirects stdout/stderr to a log file (useful when running as a packaged .exe with no console)
log_file = open("logs.txt", "a", encoding="utf-8")
sys.stdout = log_file
sys.stderr = log_file

# loads the YOLO model used to detect the tractor
model = YOLO("best.pt")
app = Flask(__name__)

# holds the most recent annotated frame, shared between the yolo_worker thread and the video_feed route
latest_frame = None

# lock to prevent reading/writing latest_frame at the same time from different threads
frame_lock = threading.Lock()

# saves a detection session (start, end, duration, image) to the database and uploads the image to storage
def guarda_bd(t_inicio, t_fim, t_total, imagem):
    try:
        # converts the captured image into bits
        _, buffer = cv2.imencode('.jpg', imagem, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        file_bits = io.BytesIO(buffer).read()

        # uploads the image to the supabase bucket and gets its public url
        file_path = f"foto_{t_inicio}.jpg"
        supabase.storage.from_(bucket_name).upload(file_path, file_bits, {"content-type": "image/jpeg"})
        url_publica = supabase.storage.from_(bucket_name).get_public_url(file_path)

        data_hoje = datetime.now().strftime('%Y-%m-%d')

        # counts today's detections to determine the next daily counter value
        response_count = supabase.table("detecoes")\
            .select("*", count="exact")\
            .filter("timestamp_inicio", "ilike", f"{data_hoje}%")\
            .execute()
        
        proximo_contador = (response_count.count or 0) + 1

        # data to insert into the DB
        data = {
            "timestamp_inicio": t_inicio,
            "timestamp_fim": t_fim,
            "duracao_total": t_total,
            "veiculo": "Trator",
            "counter_dia": proximo_contador,
            "imagem_detecao": url_publica
        }
        
        # actually inserts into the database
        supabase.table("detecoes").insert(data).execute()
        print(f"Inserido na base de dados: {t_total}s (Contador: {proximo_contador})")
    except Exception as e:
        print(f"Erro ao guardar na base de dados: {e}")

# main worker: reads camera frames, runs YOLO detection, tracks tractor presence over time
# and triggers a DB save once a full detection session ends
def yolo_worker():
    global latest_frame
    cap = cv2.VideoCapture(CAMERA_URL)
    
    # whether a tractor is currently considered "in scene" (an active session is open)
    trator_na_cena = False
    
    # real timestamp of when the current session started
    timestamp_inicio_real = None
    
    # timestamp (time.time()) of the last frame where the tractor was seen
    ultimo_momento_visto = 0
    
    # avoids repeating the "tractor lost, waiting..." log on every frame
    aviso_perda_enviado = False
    
    # timestamp of the previous processed frame, used to compute dt between frames
    ultimo_momento_frame = time.time()
    
    # total accumulated time (seconds) the tractor was actually seen during the current session
    tempo_acumulado_visto = 0
    
    # frame to be saved as the session's snapshot image
    frame_para_guardar = None
    
    # whether the snapshot for the current session has already been captured
    ja_capturou_imagem = False
    
    # how long (seconds) to wait without seeing the tractor before closing the session
    TEMPO_ESPERA_PARA_FECHAR = 30  
    
    # minimum accumulated time (seconds) the tractor must be seen for the session to be saved as a detection
    TEMPO_MINIMO_DETECAO = 6 

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            time.sleep(2)
            cap = cv2.VideoCapture(CAMERA_URL)
            continue
            
        results = model(frame, verbose=False, conf=0.9) 
        annotated_frame = results[0].plot()

       # calculates the center of the screen
        altura, largura, _ = frame.shape
        meio_do_ecra = largura // 2
        
        detetou_no_lado_correto = False

        # finds objects and calculates where the YOLO box is (along the x axis)
        for box in results[0].boxes:
            x1, y1, x2, y2= box.xyxy[0].tolist()
            centro_x = (x1 + x2) / 2
            
            # if it's before the middle of the screen, continue (class must be 0 for tractor) 
            if int(box.cls[0]) == 0 and centro_x < meio_do_ecra: # without the "and" to detect across the full screen
                detetou_no_lado_correto = True
                #detetou_trator = True
                break 

        # computes elapsed time (dt) since the last processed frame
        tempo_atual = time.time()
        dt = tempo_atual - ultimo_momento_frame
        ultimo_momento_frame = tempo_atual 

        # we already know it detected and on the correct side, accumulate the time YOLO sees the object
        if detetou_no_lado_correto: #if detetou_trator: - detects across the full screen
            ultimo_momento_visto = tempo_atual
            tempo_acumulado_visto += dt 

        # in case the tractor disappears
            if aviso_perda_enviado:
                print("O YOLO voltou a detetar o trator!", flush=True)
                aviso_perda_enviado = False
            
            # detects the tractor (marks the start of a new session)
            if not trator_na_cena:
                timestamp_inicio_real = datetime.now()
                trator_na_cena = True
                print(f"Trator detetado: {timestamp_inicio_real.strftime('%H:%M:%S')}")

            # takes a screenshot 2 seconds after the tractor appears
            if trator_na_cena and not ja_capturou_imagem:
                diff = (datetime.now() - timestamp_inicio_real).total_seconds()
                if diff >= 2.0:
                    frame_para_guardar = annotated_frame.copy()
                    ja_capturou_imagem = True
                    print("Imagem capturada.")


# checks if the time without detecting the tractor exceeds the defined waiting time without tractor (30s)
        if trator_na_cena:
            tempo_sem_ver = tempo_atual - ultimo_momento_visto

            # logs a single "waiting to close" warning once the tractor is no longer seen
            if not detetou_no_lado_correto and not aviso_perda_enviado and tempo_sem_ver < TEMPO_ESPERA_PARA_FECHAR: # remove "not detetou_no_lado_correto" to use the full screen
                print(f"Sem trator... a aguardar {TEMPO_ESPERA_PARA_FECHAR}s.", flush=True)
                aviso_perda_enviado = True

# in case the time the tractor is not seen exceeds the defined time (30s), close the session
            if tempo_sem_ver > TEMPO_ESPERA_PARA_FECHAR:

                # session end time is the last moment the tractor was actually seen, not "now"
                t_fim_dt = datetime.fromtimestamp(ultimo_momento_visto)
                tempo_totaldelta = t_fim_dt - timestamp_inicio_real
                tempo_total_segundos = tempo_totaldelta.total_seconds()

# checks if the tractor was detected for at least the previously defined minimum time (6s)
                if tempo_acumulado_visto >= TEMPO_MINIMO_DETECAO:
                    t_inicio_str = timestamp_inicio_real.strftime('%Y-%m-%d %H:%M:%S')
                    t_fim_str = t_fim_dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # saves to the database
                    guarda_bd(t_inicio_str, t_fim_str, round(tempo_total_segundos, 2), frame_para_guardar)
                    print(f"Sessão guardada: {round(tempo_total_segundos, 2)}s (Visto: {round(tempo_acumulado_visto, 2)}s)")
                else:
                    # session too short (tractor barely visible) - discarded, not saved to DB
                    print(f"Não registado. Tempo insuficiente: Visto {round(tempo_acumulado_visto, 2)}s de um total de {round(tempo_total_segundos, 2)}s")
                
                # resets session state so a new detection can start fresh
                trator_na_cena = False
                timestamp_inicio_real = None
                tempo_acumulado_visto = 0.0
                aviso_perda_enviado = False
                frame_para_guardar = None
                ja_capturou_imagem = False
        
        # updates the shared frame for the /video_feed route to consume
        with frame_lock:
            latest_frame = cv2.resize(annotated_frame, (800, 450))

# thread to run the frame acquisition and detection loop
threading.Thread(target=yolo_worker, daemon=True).start()

######################################### Flask API #############################################

# route to show the live video (MJPEG stream)
@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if latest_frame is None: continue
                # converts the frame (numpy array) into JPEG-formatted bytes
                _, buffer = cv2.imencode('.jpg', latest_frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


# / route to display the dashboard home page
@app.route('/', methods= ['GET'])
def index():
    # filter to update the day of detections shown on the dashboard
    data_filtro = request.args.get('data')
    hoje_cpu = datetime.now().strftime('%Y-%m-%d')
    
    try:
        query = supabase.table("detecoes").select("*")

        # counts the day total for the filter entered by the user
        if data_filtro:
            response = query.filter("timestamp_inicio", "ilike", f"{data_filtro}%")\
                            .order("counter_dia", desc=True).execute()
            dados = response.data
            total_real = len(dados)
        else:
            # counts the day total for the current day filter (not selected by the user)
            response = query.filter("timestamp_inicio", "ilike", f"{hoje_cpu}%")\
                            .order("counter_dia", desc=True).execute()
            dados = response.data
            total_real = len(dados)
            
            dados = dados[:20] 

    except Exception as e:
        print(f"Erro ao processar dados: {e}")
        dados = []
        total_real = 0
        
    # sends data for the HTML to display
    return render_template('index.html', 
                           detecoes=dados, 
                           data_selecionada=data_filtro or hoje_cpu, 
                           total=total_real)

# route to refresh the dynamic table
@app.route('/tabela_atualizada', methods=['GET'])
def tabela_atualizada():
    # applies the date filter
    data_filtro = request.args.get('data')
    hoje_cpu = datetime.now().strftime('%Y-%m-%d')
    alvo = data_filtro if data_filtro else hoje_cpu

    try:
        # fetches the detection data
        response = supabase.table("detecoes")\
            .select("*")\
            .filter("timestamp_inicio", "ilike", f"{alvo}%")\
            .order("counter_dia", desc=True).execute()
        
        dados = response.data
        total_real = len(dados)
        exibir_dados = dados
    except Exception as e:
        exibir_dados = [], 0

    # sends data to the HTML
    return render_template('tabela_parcial.html', detecoes=exibir_dados, total=total_real, data_selecionada=alvo)

# route for the app shutdown button
@app.route('/encerrarApp', methods=['POST'])
def encerraApp():
    os._exit(0)
    print("App foi encerrada!")
    return 'App foi encerrada', 204

# opens the default browser at the app's local address
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == '__main__':
    # automatically opens the browser when the app starts, with a 2 second timer
    Timer(2, open_browser).start()
    # runs on localhost, port 5000
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)