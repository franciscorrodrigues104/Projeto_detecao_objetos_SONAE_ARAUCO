from ultralytics import YOLO
import cv2
import numpy as np

# ==================================================
# CONFIGURAÇÃO
# ==================================================

MODELO_MANTA = r"C:\Users\frrodrigues.INDPT\Desktop\projeto_defeitos_manta\passo_2_teste_modelo\modelo_seg_manta_v2.pt"

MODELO_DEFEITOS = r"C:\Users\frrodrigues.INDPT\Desktop\projeto_defeitos_manta\passo_2_teste_modelo\modelo_detecao_defeitos_manta_v2.pt"

VIDEO_INPUT = r"C:\Users\frrodrigues.INDPT\Desktop\projeto_defeitos_manta\passo_1_extracao_imagens\Export_2026-09-08_13_50_50_220\P068 – LC – Manta de Formação 2026-09-06_05_17_54_384.mkv"

VIDEO_OUTPUT = r"C:\Users\frrodrigues.INDPT\Desktop\projeto_defeitos_manta\passo_2_teste_modelo\resultado_videos\dashboard_defeitos_3.mp4"

CONF_MANTA = 0.7
CONF_DEFEITOS = 0.4

NUM_ZONAS = 20

# ==================================================
# MODELOS
# ==================================================

modelo_manta = YOLO(MODELO_MANTA)
modelo_defeitos = YOLO(MODELO_DEFEITOS)

# ==================================================
# HISTOGRAMA
# ==================================================

histograma = np.zeros(
    NUM_ZONAS,
    dtype=np.int32
)

ids_contados = set()

# ==================================================
# VIDEO
# ==================================================

cap = cv2.VideoCapture(VIDEO_INPUT)

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

writer = cv2.VideoWriter(
    VIDEO_OUTPUT,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width * 3, height)
)

frame_count = 0

# ==================================================
# LOOP
# ==================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # ===============================================
    # SEGMENTAÇÃO DA MANTA
    # ===============================================

    resultado_manta = modelo_manta(
        frame,
        conf=CONF_MANTA,
        verbose=False
    )[0]

    frame_segmentado = resultado_manta.plot()

    painel_roi = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    # ===============================================
    # ROI DA MANTA
    # ===============================================

    if resultado_manta.masks is not None:

        maior_area = 0
        melhor_polygon = None

        for polygon in resultado_manta.masks.xy:

            polygon = polygon.astype(
                np.int32
            )

            area = cv2.contourArea(
                polygon
            )

            if area > maior_area:

                maior_area = area
                melhor_polygon = polygon

        if melhor_polygon is not None:

            # ======================================
            # CRIAR MÁSCARA DA MANTA
            # ======================================

            mask = np.zeros(
                frame.shape[:2],
                dtype=np.uint8
            )

            cv2.fillPoly(
                mask,
                [melhor_polygon],
                255
            )

            # ======================================
            # REDUZIR A MÁSCARA
            # REMOVE AS BORDAS AZUIS
            # ======================================

            kernel = np.ones(
                (2,2),
                np.uint8
            )

            mask = cv2.erode(
                mask,
                kernel,
                iterations=1
            )

            # ======================================
            # APLICAR MÁSCARA AO FRAME
            # ======================================

            frame_mascarado = cv2.bitwise_and(
                frame,
                frame,
                mask=mask
            )

            # ======================================
            # BOUNDING BOX DA MANTA
            # ======================================

            x, y, w, h = cv2.boundingRect(
                melhor_polygon
            )

            # ======================================
            # RECORTAR ROI
            # ======================================

            roi = frame_mascarado[
                y:y+h,
                x:x+w
            ]

            mask_roi = mask[
                y:y+h,
                x:x+w
            ]

            roi = cv2.bitwise_and(
                roi,
                roi,
                mask=mask_roi
            )

            # ======================================
            # DEBUG
            # ======================================

            if frame_count == 100:
                cv2.imwrite(
                    "debug_roi.png",
                    roi
                )

                        # ======================================
            # DETEÇÃO DE DEFEITOS
            # ======================================

            if roi.size > 0:

                resultado = modelo_defeitos.track(
                    roi,
                    conf=CONF_DEFEITOS,
                    persist=True,
                    verbose=False
                )[0]

                # imagem base para desenhar (sem usar .plot(),
                # porque .plot() desenha TODAS as deteções, mesmo
                # as que vamos filtrar a seguir)
                roi_anotada = roi.copy()

                defeitos_visiveis = 0

                if (
                    resultado.boxes is not None and
                    resultado.boxes.id is not None
                ):

                    boxes = resultado.boxes.xyxy.cpu().numpy()
                    ids = resultado.boxes.id.cpu().numpy().astype(int)
                    confs = resultado.boxes.conf.cpu().numpy()
                    classes = resultado.boxes.cls.cpu().numpy().astype(int)
                    nomes = resultado.names

                    LIMIAR_SOBREPOSICAO_MANTA = 0.6  # ajusta se necessário

                    for box, track_id, conf, cls in zip(
                        boxes, ids, confs, classes
                    ):

                        x1, y1, x2, y2 = box.astype(int)

                        x1c = max(x1, 0)
                        y1c = max(y1, 0)
                        x2c = min(x2, mask_roi.shape[1])
                        y2c = min(y2, mask_roi.shape[0])

                        if x2c <= x1c or y2c <= y1c:
                            continue

                        regiao_mask = mask_roi[y1c:y2c, x1c:x2c]

                        if regiao_mask.size == 0:
                            continue

                        proporcao_manta = (
                            np.count_nonzero(regiao_mask) /
                            regiao_mask.size
                        )

                        # descarta deteções que caem
                        # maioritariamente fora da manta
                        if proporcao_manta < LIMIAR_SOBREPOSICAO_MANTA:
                            continue

                        defeitos_visiveis += 1

                        # contagem única
                        if track_id not in ids_contados:

                            ids_contados.add(track_id)

                            centro_x = (x1 + x2) / 2
                            pos_norm = centro_x / roi.shape[1]
                            zona = int(pos_norm * NUM_ZONAS)
                            zona = min(zona, NUM_ZONAS - 1)

                            histograma[zona] += 1

                        # desenhar manualmente apenas as deteções válidas
                        cv2.rectangle(
                            roi_anotada,
                            (x1, y1),
                            (x2, y2),
                            (255, 128, 0),
                            2
                        )

                        rotulo = f"{nomes[cls]} {conf:.2f}"

                        cv2.putText(
                            roi_anotada,
                            rotulo,
                            (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 128, 0),
                            2
                        )

                cv2.putText(
                    roi_anotada,
                    f"Defeitos visiveis: {defeitos_visiveis}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    roi_anotada,
                    f"Defeitos unicos: {len(ids_contados)}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    roi_anotada,
                    "Background preto",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2
                )

                painel_roi = cv2.resize(
                    roi_anotada,
                    (width, height)
                )

    # ===============================================
    # HISTOGRAMA
    # ===============================================

    painel_hist = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    max_val = max(
        histograma.max(),
        1
    )

    largura_barra = width // NUM_ZONAS

    for i in range(NUM_ZONAS):

        valor = histograma[i]

        altura = int(
            (
                valor /
                max_val
            )
            * (height - 180)
        )

        x1 = i * largura_barra

        x2 = (
            x1 +
            largura_barra - 4
        )

        y1 = (
            height -
            altura -
            50
        )

        y2 = height - 50

        cv2.rectangle(
            painel_hist,
            (x1, y1),
            (x2, y2),
            (0, 165, 255),
            -1
        )

        cv2.putText(
            painel_hist,
            str(i + 1),
            (x1 + 3, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

    cv2.putText(
        painel_hist,
        "Distribuicao de Defeitos",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        painel_hist,
        "Direita -> Esquerda",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (200, 200, 200),
        2
    )

    cv2.putText(
        painel_hist,
        f"Defeitos unicos: {len(ids_contados)}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # ===============================================
    # TITULOS
    # ===============================================

    cv2.putText(
        frame_segmentado,
        "Segmentacao da Manta",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        painel_roi,
        "ROI + Tracking de Defeitos",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # ===============================================
    # DASHBOARD FINAL
    # ===============================================

    frame_final = np.hstack([
        frame_segmentado,
        painel_roi,
        painel_hist
    ])

    writer.write(
        frame_final
    )

    if frame_count % 100 == 0:

        print(
            f"Frames processados: {frame_count}"
        )

# ==================================================
# FECHAR
# ==================================================

cap.release()
writer.release()

print()
print("====================================")
print("RESUMO FINAL")
print("====================================")

print(
    f"Frames processados: {frame_count}"
)

print(
    f"Defeitos unicos: {len(ids_contados)}"
)

print()
print("Distribuicao por zona:")

for i, valor in enumerate(histograma):

    print(
        f"Zona {i+1}: {valor}"
    )

print()
print("Video:")
print(VIDEO_OUTPUT)