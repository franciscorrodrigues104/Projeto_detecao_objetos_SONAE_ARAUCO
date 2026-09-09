# Portfolio de Estágio Curricular - Sonae Arauco

**Autor:** Francisco Ribeiro Rodrigues  
**Instituição:** Sonae Arauco (Estágio Curricular)

Este repositório documenta os projetos de visão computacional desenvolvidos durante o meu estágio na Sonae Arauco, focados na automatização da monitorização industrial através de inteligência artificial.

---

## Índice
1. [Projeto 1: Deteção e Contagem de Tratores](#projeto-1)
2. [Projeto 2: Deteção de Falhas em Lotes](#projeto-2)
3. [Tecnologias Utilizadas](#tecnologias-utilizadas)

---

## Projeto 1: Deteção e Contagem de Tratores

Sistema desenvolvido para a automação da monitorização logística de tratores em ambiente industrial.

### Funcionalidades
* **Deteção de Alta Precisão:** Identificação de tratores com métricas de sucesso superiores a 99%.
* **Modelo Finalizado:** YOLOv8 treinado especificamente para o ambiente fabril da Sonae Arauco.
* **Interface Web:** Dashboard em Flask para visualização das deteções e contagem automática.
* **Integração Cloud:** Registo de dados no Supabase para análise histórica e relatórios.

### Performance e Resultados Finais
O treino foi concluído após 100 épocas (aprox. 3.7 horas de processamento):

* **mAP50:** 0.995 (99.5%)
* **mAP50-95:** 0.99 (99%)
* **Precisão (P):** 0.974
* **Recall (R):** 1.0
* **Velocidade de Inferência:** ~66.3ms por imagem

---

## Projeto 2: Deteção de Falhas em Lotes

Sistema especializado na deteção automática de anomalias em lotes de produção em tempo real.

### Funcionalidades
* **Monitorização em Tempo Real:** Deteção contínua de falhas na linha de produção.
* **Mecanismo de Estabilidade:** Aplicação de um cooldown de 6 segundos para evitar registos duplicados ou ruído.
* **Dashboard Dinâmico:** Visualização de alertas e histórico de falhas com atualização automática.

### Performance e Resultados Finais
O modelo atingiu os seguintes níveis de confiança após o treino:

**Modelo de Deteção e Contagem de Tratores:**
* **mAP50:** 0.995 (99.5%)
* **mAP50-95:** 0.99 (99%)
* **Precisão (P):** 0.974
* **Recall (R):** 1.0
* **Velocidade de Inferência:** ~66.3ms por imagem

**Modelo de Deteção de Falhas em Lotes (Desalinhamento e Falhas):**
* **mAP50:** 0.534 (53.4%)
* **mAP50-95:** 0.188 (18.8%)
* **Precisão (P):** 0.763
* **Recall (R):** 0.509
* **Velocidade de Inferência:** ~5.8ms por imagem (preprocessamento + inferência)
---

## Tecnologias Utilizadas

* **IA / Deep Learning:** YOLOv8/YOLO11s (Ultralytics), PyTorch
* **Linguagem:** Python 3.12
* **Backend:** Flask & Supabase (PostgreSQL)
* **Interface:** HTML, JavaScript, HTMX
* **Hardware de Teste:** Intel Core i5-8250U @ 1.60GHz

---

## Instalação e Execução

1. **Clonar o repositório:**
   ```bash
   git clone [https://github.com/franciscorrodrigues104/Projeto_detecao_objetos_SONAE_ARAUCO.git](https://github.com/franciscorrodrigues104/Projeto_detecao_objetos_SONAE_ARAUCO.git)
