from ultralytics import YOLO

model = YOLO('yolov8n.pt') 

results = model.train(
    data=r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_3_treino_modelo\data.yaml", #ficheiro yaml
    epochs=50,       #nº de vezes a ser estudado
    imgsz=640,      #tamanho das imagens
)

print("Treino concluído!")