import requests
from supabase import create_client
from concurrent.futures import ThreadPoolExecutor
import os

# Configuração
supabase = create_client("https://rqxkyqmbigutmrftjclh.supabase.co", "sb_secret_KEnpyvmtnl-38Vfx6yaqgg_UwDHL9-D")
BUCKET_NAME = 'fotos_detecao_falhas'

def baixar_imagem(caminho):
    try:
        # Gerar URL assinado (para buckets privados)
        res = supabase.storage.from_(BUCKET_NAME).create_signed_url(caminho, 3600)
        url = res['signedURL']
        
        # Download
        r = requests.get(url)
        if r.status_code == 200:
            # Garante que o nome do ficheiro é salvo corretamente
            nome_final = caminho.split('/')[-1]
            with open(f"pasta_destino/{nome_final}", 'wb') as f:
                f.write(r.content)
            return f"Sucesso: {caminho}"
        return f"Falha (Status {r.status_code}): {caminho}"
    except Exception as e:
        return f"Erro no download de {caminho}: {e}"

# 1. Obter a lista da base de dados
response = supabase.table('detecoes_falhas_duplicate').select('imagem_original').gte('timestamp_inicio', '2026-07-17T00:00:00').execute()
lista_imagens = [item['imagem_original'] for item in response.data]

# 2. Executar downloads em paralelo
if lista_imagens:
    with ThreadPoolExecutor(max_workers=10) as executor:
        resultados = list(executor.map(baixar_imagem, lista_imagens))
    
    for resultado in resultados:
        print(resultado)
else:
    print("Nenhuma imagem encontrada na base de dados para a data especificada.")