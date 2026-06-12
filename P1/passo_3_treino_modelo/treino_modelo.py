from ultralytics import YOLO

model = YOLO('yolov8n.pt') 

results = model.train(
    data=r"C:\Users\frrodrigues\Desktop\Projeto_detecao_trator\passo_3_treino_modelo\data.yaml", #ficheiro yaml
    epochs=100,       #nº de vezes a ser estudado
    imgsz=640,      #tamanho das imagens
)

print("Treino concluído!")