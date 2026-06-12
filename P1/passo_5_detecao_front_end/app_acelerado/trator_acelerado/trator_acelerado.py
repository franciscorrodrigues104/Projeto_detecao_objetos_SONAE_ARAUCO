from flask import Flask, render_template, Response, request, jsonify
import cv2
from ultralytics import YOLO
import os
import json
from datetime import datetime, timedelta
import glob
import webbrowser
from threading import Timer
import sys

app = Flask(__name__)

# Configurações fixas
STRIDE = 30
OUTPUT_FOLDER = "PrintScreens_detecoes"
TEMPO_ESPERA_PARA_FECHAR = 30
TEMPO_MINIMO_DETECAO = 6


if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

if getattr(sys, 'frozen', False):
    template_folder = os.path.join(sys._MEIPASS, 'templates')
    app = Flask(__name__, template_folder=template_folder)
else:
    app = Flask(__name__)

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

MODEL_PATH = resource_path("best.pt")


def encontrar_video_na_pasta():
    """Procura o primeiro ficheiro de vídeo disponível na pasta do script."""
    extensoes = ['*.mp4', '*.mkv', '*.avi', '*.mov', '*.webm']
    for ext in extensoes:
        videos = glob.glob(ext)
        if videos:
            return videos[0] 
    return None

inProcessing = False
parar_processamento = False

def processar_video_logic(data_inicio_str):
    global inProcessing  
    global parar_processamento
    parar_processamento = False
    inProcessing = True
    inicio_dt = datetime.strptime(data_inicio_str, "%Y-%m-%d %H:%M:%S")

    video_input = encontrar_video_na_pasta()
    
    if not video_input:
        yield f"data: {json.dumps({'log': 'Erro: Nenhum vídeo (.mp4, .mkv, .avi, .webm) encontrado na pasta!'})}\n\n"
        return

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(video_input)
    
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_iteracoes = total_frames // STRIDE
    largura_video = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    meio_do_ecra = largura_video // 2

    tratores_detetados_total = 0
    trator_na_cena = False
    timestamp_inicio_evento = 0
    ultimo_momento_visto = 0
    tempo_acumulado_visto = 0
    ja_capturou_imagem = False
    ultimo_print_path = ""
    ultimo_segundo_processado = 0

    results = model.predict(source=video_input, stream=True, verbose=False, imgsz=320, vid_stride=STRIDE, conf=0.9)

    for frame_idx, r in enumerate(results):

        if parar_processamento:
            yield f"data: {json.dumps({'log': 'Processamento interrompido!', 'done': True})}\n\n"
            break

        segundos_no_video = (frame_idx * STRIDE) / fps_video
        dt = segundos_no_video - ultimo_segundo_processado
        ultimo_segundo_processado = segundos_no_video

        detetou_no_lado_correto = False
        for box in r.boxes:
            if box.xyxy.shape[0] > 0:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                centro_x = (x1 + x2) / 2
                if int(box.cls[0]) == 0 and centro_x < meio_do_ecra:
                    detetou_no_lado_correto = True
                    break

        if detetou_no_lado_correto:
            ultimo_momento_visto = segundos_no_video
            tempo_acumulado_visto += dt
            if not trator_na_cena:
                trator_na_cena = True
                timestamp_inicio_evento = segundos_no_video
                tratores_detetados_total += 1
                ja_capturou_imagem = False
                yield f"data: {json.dumps({'log': f'Trator detetado!'})}\n\n"

            if not ja_capturou_imagem and (segundos_no_video - timestamp_inicio_evento) >= 2.0:
                hora_real = inicio_dt + timedelta(seconds=segundos_no_video)
                nome_foto = f"detecao_{hora_real.strftime('%H-%M-%S')}.jpg"
                ultimo_print_path = os.path.join(OUTPUT_FOLDER, nome_foto)  
                cv2.imwrite(ultimo_print_path, r.plot())
                yield f"data: {json.dumps({'log': f'Trator detetado às {hora_real.strftime('%H-%M-%S')}'})}\n\n"
                ja_capturou_imagem = True

        if trator_na_cena:
            if (segundos_no_video - ultimo_momento_visto) > TEMPO_ESPERA_PARA_FECHAR:
                if tempo_acumulado_visto >= TEMPO_MINIMO_DETECAO:
                    yield f"data: {json.dumps({'log': f'Detecao {tratores_detetados_total} guardada como PrintScreen! - {nome_foto}'})}\n\n"
                else:
                    if os.path.exists(ultimo_print_path): os.remove(ultimo_print_path)
                    tratores_detetados_total -= 1
                    yield f"data: {json.dumps({'log': f'Detecao descartada (tempo insuficiente)'})}\n\n"
                trator_na_cena = False
                tempo_acumulado_visto = 0

        percentagem = int((frame_idx / total_iteracoes) * 100)
        yield f"data: {json.dumps({'percent': percentagem, 'total': tratores_detetados_total})}\n\n"

    cap.release()
    yield f"data: {json.dumps({'percent': 100, 'done': True})}\n\n"
    inProcessing = False

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/progresso')
def progress():
    data_inicio = request.args.get('inicio', '2026-01-01 00:00:00')
    return Response(processar_video_logic(data_inicio), mimetype='text/event-stream')



@app.route('/encerrarApp', methods=['POST'])
def encerraApp():
    os._exit(0)
    return '', 204


@app.route('/parar_agora', methods=['POST'])
def parar_agora():

    global parar_processamento
    parar_processamento = True
    return '', 204

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == '__main__':
    Timer(2, open_browser).start()
    app.run(debug=False, port=5000, use_reloader=False)