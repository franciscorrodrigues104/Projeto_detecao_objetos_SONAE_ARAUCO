from ultralytics import YOLO
import cv2
import numpy as np
import os
from glob import glob

# ==================================================
# CONFIGURAÇÃO
# ==================================================

MODELO = r"C:\Users\frrodrigues.INDPT\Desktop\projeto_defeitos_manta\passo_2_teste_modelo\modelo_seg_manta_v2.pt"

PASTA_IMAGENS = r"C:\Users\frrodrigues.INDPT\Desktop\projeto_defeitos_manta\passo_1_extracao_imagens\novos_frames_defeitos"

PASTA_SAIDA = r"C:\Users\frrodrigues.INDPT\Desktop\projeto_defeitos_manta\passo_1_extracao_imagens\dataset_defeitos_v2"

CONF = 0.1

# ==================================================
# CRIAR PASTA DE SAÍDA
# ==================================================

os.makedirs(PASTA_SAIDA, exist_ok=True)

# ==================================================
# CARREGAR MODELO
# ==================================================

model = YOLO(MODELO)

# ==================================================
# LISTAR IMAGENS
# ==================================================

extensoes = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

imagens = []

for ext in extensoes:
    imagens.extend(glob(os.path.join(PASTA_IMAGENS, ext)))

print(f"\nEncontradas {len(imagens)} imagens\n")

# ==================================================
# PROCESSAMENTO
# ==================================================

for i, imagem_path in enumerate(imagens):

    try:

        print(f"[{i+1}/{len(imagens)}] {os.path.basename(imagem_path)}")

        # -----------------------------
        # Inferência
        # -----------------------------
        results = model(
            imagem_path,
            conf=CONF,
            verbose=False
        )

        result = results[0]

        # -----------------------------
        # Ler imagem original
        # -----------------------------
        img = cv2.imread(imagem_path)

        if img is None:
            print("Erro ao abrir imagem.")
            continue

        # -----------------------------
        # Sem deteção
        # -----------------------------
        if result.masks is None:
            print("Nenhuma manta encontrada.")
            continue

        # -----------------------------
        # Selecionar a maior máscara
        # (caso apareçam várias)
        # -----------------------------
        maior_area = 0
        melhor_polygon = None

        for polygon in result.masks.xy:

            polygon = polygon.astype(np.int32)

            area = cv2.contourArea(polygon)

            if area > maior_area:
                maior_area = area
                melhor_polygon = polygon

        if melhor_polygon is None:
            continue

        # -----------------------------
        # Máscara binária
        # -----------------------------
        mask = np.zeros(
            img.shape[:2],
            dtype=np.uint8
        )

        cv2.fillPoly(
            mask,
            [melhor_polygon],
            255
        )

        # -----------------------------
        # Aplicar máscara
        # -----------------------------
        segmented = cv2.bitwise_and(
            img,
            img,
            mask=mask
        )

        # -----------------------------
        # ROI da manta
        # -----------------------------
        x, y, w, h = cv2.boundingRect(
            melhor_polygon
        )

        crop = segmented[
            y:y+h,
            x:x+w
        ]

        # -----------------------------
        # Guardar
        # -----------------------------
        nome = os.path.splitext(
            os.path.basename(imagem_path)
        )[0]

        caminho_saida = os.path.join(
            PASTA_SAIDA,
            f"{nome}_roi.png"
        )

        cv2.imwrite(
            caminho_saida,
            crop
        )

    except Exception as e:

        print(
            f"Erro em {imagem_path}"
        )
        print(e)

print("\nProcessamento terminado!")