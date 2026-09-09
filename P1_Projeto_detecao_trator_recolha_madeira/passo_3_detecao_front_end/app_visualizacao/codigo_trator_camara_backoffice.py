import cv2
from flask import Flask, Response, render_template, redirect, request
from ultralytics import YOLO
import time
from datetime import datetime
import os
import webbrowser
from threading import Timer
import sys
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

MODEL_PATH = resource_path("best.pt")
model = YOLO(MODEL_PATH)

app = Flask(__name__, template_folder=resource_path("templates"), static_folder=resource_path("static"))


def gen_frames():
    cap = cv2.VideoCapture(CAMERA_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            time.sleep(2)
            cap = cv2.VideoCapture(CAMERA_URL)
            continue

        results = model(frame, verbose=False, conf=0.9) 
        annotated_frame = results[0].plot()

        detetou_neste_frame = any(int(box.cls[0]) == 0 for box in results[0].boxes)

        if detetou_neste_frame:

            display_frame = cv2.resize(annotated_frame, (800, 450))
        else:
            display_frame = cv2.resize(frame, (800, 450))

        ret, buffer = cv2.imencode('.jpg', display_frame)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
    

    cap.release()


@app.route('/', methods = ['GET'])
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

@app.route('/tabela_atualizada', methods = ['GET'])
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
        exibir_dados, total = [], 0

    return render_template('tabela_parcial.html', detecoes=exibir_dados, total=total_real, data_selecionada=alvo)

@app.route('/video_feed', methods = ['GET'])
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

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