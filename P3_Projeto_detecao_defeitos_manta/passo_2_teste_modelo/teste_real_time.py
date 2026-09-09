from ultralytics import YOLO
import cv2
import numpy as np

# ==================================================
# CONFIGURAÇÃO
# ==================================================

MODELO_MANTA = r"modelo_seg_manta_v2.pt"
MODELO_DEFEITOS = r"modelo_defeitos_manta_v1.pt"

URL_MJPG = "https://Leo:Xonae$2410@10.13.45.68/axis-cgi/mjpg/video.cgi"

CONF_MANTA = 0.7
CONF_DEFEITOS = 0.5

# ==================================================
# MODELOS
# ==================================================

modelo_manta = YOLO(MODELO_MANTA)
modelo_defeitos = YOLO(MODELO_DEFEITOS)

# ==================================================
# TRACKING
# ==================================================

ids_contados = set()

# ==================================================
# STREAM MJPG
# ==================================================

cap = cv2.VideoCapture(URL_MJPG)

if not cap.isOpened():
    raise Exception("Erro ao abrir stream MJPG")

# ==================================================
# LOOP
# ==================================================

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    # =============================================
    # SEGMENTAÇÃO DA MANTA
    # =============================================

    resultado_manta = modelo_manta(
        frame,
        conf=CONF_MANTA,
        verbose=False
    )[0]

    frame_segmentado = resultado_manta.plot()

    painel_roi = np.zeros_like(frame)

    # =============================================
    # ENCONTRAR MAIOR MÁSCARA
    # =============================================

    if resultado_manta.masks is not None:

        maior_area = 0
        melhor_polygon = None

        for polygon in resultado_manta.masks.xy:

            polygon = polygon.astype(np.int32)

            area = cv2.contourArea(polygon)

            if area > maior_area:
                maior_area = area
                melhor_polygon = polygon

        if melhor_polygon is not None:

            # =====================================
            # MÁSCARA
            # =====================================

            mask = np.zeros(
                frame.shape[:2],
                dtype=np.uint8
            )

            cv2.fillPoly(
                mask,
                [melhor_polygon],
                255
            )

            # reduzir bordas

            kernel = np.ones((2, 2), np.uint8)

            mask = cv2.erode(
                mask,
                kernel,
                iterations=1
            )

            frame_mascarado = cv2.bitwise_and(
                frame,
                frame,
                mask=mask
            )

            x, y, w, h = cv2.boundingRect(
                melhor_polygon
            )

            roi = frame_mascarado[
                y:y+h,
                x:x+w
            ]

            if roi.size > 0:

                # =====================================
                # TRACKING DEFEITOS
                # =====================================

                resultado_defeitos = modelo_defeitos.track(
                    roi,
                    conf=CONF_DEFEITOS,
                    persist=True,
                    verbose=False
                )[0]

                roi_anotada = resultado_defeitos.plot()

                if (
                    resultado_defeitos.boxes is not None and
                    resultado_defeitos.boxes.id is not None
                ):

                    ids = (
                        resultado_defeitos.boxes.id
                        .cpu()
                        .numpy()
                        .astype(int)
                    )

                    for track_id in ids:
                        ids_contados.add(track_id)

                cv2.putText(
                    roi_anotada,
                    f"Visiveis: {len(resultado_defeitos.boxes)}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    roi_anotada,
                    f"Unicos: {len(ids_contados)}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                painel_roi = cv2.resize(
                    roi_anotada,
                    (frame.shape[1], frame.shape[0])
                )

    # =============================================
    # DASHBOARD
    # =============================================

    
    dashboard = np.hstack([
        frame_segmentado,
        painel_roi
    ])

    cv2.imshow(
        "Dashboard Defeitos",
        dashboard
    )

    tecla = cv2.waitKey(1)

    if tecla == 27:
        break

# ==================================================
# FECHAR
# ==================================================

cap.release()
cv2.destroyAllWindows()

print(
    f"Defeitos unicos: {len(ids_contados)}"
)