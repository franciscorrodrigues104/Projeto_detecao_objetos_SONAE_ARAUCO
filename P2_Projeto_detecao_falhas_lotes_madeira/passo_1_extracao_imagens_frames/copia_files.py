import os
import shutil
import re

# Lista de números que pretendes mover
numeros_alvo = {
    119
,138
,166
,238
,245
,253
,264
,280
,282
,293
,313
,321
,352
,377
,385
,392
,398
,404
,425
,426
,433
,438
,445
,455
,461
,499
,507
,512
,517
,522
,528
,534
,540
,592
,597
,601
,602
,609
,620
,626
,676
,681
,687
,698
,706
,717
,723
,729
,739
,749
}

# Converter para conjunto (set) de strings para facilitar a comparação exata
numeros_alvo_str = {str(n) for n in numeros_alvo}

# Caminhos de origem e destino
pasta_origem = r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_1_extracao_imagens_frames\runs\detect\predict9\crops\lotes"
pasta_destino = r"C:\Users\frrodrigues\Desktop\Projeto_falhas_placas\passo_1_extracao_imagens_frames\dataset_falhas_cores_p124"

# Criar a pasta de destino caso não exista
os.makedirs(pasta_destino, exist_ok=True)

contador = 0
nao_encontrados = []

# Percorrer os ficheiros da pasta de origem
for ficheiro in os.listdir(pasta_origem):
    # Usar regex para extrair o número de dentro dos parênteses, ex: falhas_cores (28).jpg -> 28
    match = re.search(r'\((\d+)\)', ficheiro)
    
    if match:
        numero_extraido = match.group(1)
        
        # Verificar se este número está na nossa lista alvo
        if numero_extraido in numeros_alvo_str:
            caminho_origem = os.path.join(pasta_origem, ficheiro)
            caminho_destino = os.path.join(pasta_destino, ficheiro)
            
            # Garantir que é um ficheiro e não uma pasta
            if os.path.isfile(caminho_origem):
                shutil.move(caminho_origem, caminho_destino)
                contador += 1
                # Remover do set para sabermos quais é que eventualmente faltaram
                numeros_alvo_str.remove(numero_extraido)

print(f"Processo concluído! Foram movidas {contador} imagens para a pasta de destino.")

if numeros_alvo_str:
    print(f"\nAviso: Os seguintes números da lista não foram encontrados na pasta: {sorted([int(x) for x in numeros_alvo_str])}")