from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO(r"C:\Users\frrodrigues.INDPT\Desktop\projeto_defeitos_manta\passo_2_teste_modelo\modelo_seg_manta_v2.pt")

results = model(
    r"C:\Users\frrodrigues.INDPT\Pictures\s.jpg",
    conf=0.1,
    verbose=False
)

result = results[0]

img = cv2.imread( r"C:\Users\frrodrigues.INDPT\Pictures\s.jpg")

polygon = result.masks.xy[0].astype(np.int32)

mask = np.zeros(img.shape[:2], dtype=np.uint8)

cv2.fillPoly(mask, [polygon], 255)

segmented = cv2.bitwise_and(img, img, mask=mask)

x, y, w, h = cv2.boundingRect(polygon)

crop = segmented[y:y+h, x:x+w]

cv2.imwrite("manta_roi_3.png", crop)

print("ROI guardada!")