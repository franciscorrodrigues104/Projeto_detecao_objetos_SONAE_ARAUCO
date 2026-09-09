# Portfolio de Estágio Curricular - Sonae Arauco

**Autor:** Francisco Ribeiro Rodrigues  
**Instituição:** Sonae Arauco (Estágio Curricular)

Este repositório documenta os projetos de visão computacional desenvolvidos durante o meu estágio na Sonae Arauco, focados na automatização da monitorização industrial através de inteligência artificial.

---

## Índice

1. [Projeto-1
2. [Projeto 2: Deteção de Falhas em Lotes](#projeto-2eitos em Manta Contínua](#pro e Contagem de Tratores

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

## Projeto 3: Deteção e Análise de Defeitos em Manta Contínua

Sistema desenvolvido para a monitorização automática da qualidade de mantas em linha de produção, recorrendo a modelos de Inteligência Artificial para segmentação, deteção de defeitos e análise estatística em tempo real.

### Funcionalidades

* **Segmentação da Manta:** Utilização de um modelo YOLO de segmentação para identificar a área útil da manta em tempo real.
* **Deteção de Defeitos:** Aplicação de um segundo modelo YOLO treinado especificamente para identificar diferentes tipos de defeitos na superfície da manta.
* **Tracking Inteligente:** Integração do algoritmo ByteTrack para evitar contagens duplicadas e acompanhar defeitos ao longo da sequência de vídeo.
* **Dashboard Web em Tempo Real:** Visualização simultânea do vídeo processado, estatísticas e indicadores operacionais.
* **Heatmap de Defeitos:** Distribuição dos defeitos por zonas da manta para identificação de padrões recorrentes na produção.
* **Histograma por Classe:** Contagem e análise da frequência de ocorrência de cada tipo de defeito.
* **Tendência Temporal:** Visualização da evolução horária dos defeitos registados.
* **Armazenamento Automático:** Registo de deteções em ficheiros CSV e armazenamento das respetivas imagens para análise posterior.
* **Monitorização de Performance:** Indicadores em tempo real de FPS da câmara, FPS de inferência, latência média e estado do sistema.
* **Recuperação Automática de Falhas:** Reconexão automática à câmara em caso de perda temporária de comunicação.

### Arquitetura da Solução

O sistema foi desenvolvido com uma arquitetura multithread otimizada para processamento contínuo:

* **Thread de Captura:** Responsável pela aquisição do stream de vídeo da linha de produção.
* **Thread de Inferência:** Executa a segmentação da manta, deteção de defeitos e tracking.
* **Thread de Escrita:** Guarda imagens e registos sem impactar o desempenho da inferência.
* **Thread de Monitorização:** Responsável pela recolha de métricas de desempenho e estado do sistema.

### Principais Resultados

* Automatização da inspeção visual da manta em tempo real.
* Redução da necessidade de inspeção manual contínua.
* Eliminação de duplicados através de tracking por identificador único.
* Criação de histórico de defeitos para suporte à análise de qualidade.
* Dashboard interativo para apoio à tomada de decisão operacional.
* Sistema preparado para funcionamento contínuo em ambiente industrial.

---

## Tecnologias Utilizadas

* **IA / Deep Learning:** YOLOv8/YOLO11s (Ultralytics), PyTorch
* **Visão Computacional:** OpenCV
* **Tracking de Objetos:** ByteTrack
* **Linguagem:** Python 3.12
* **Backend:** Flask & Supabase (PostgreSQL)
* **Interface:** HTML, JavaScript, HTMX
* **Processamento Numérico:** NumPy
* **Hardware de Teste:** Intel Core i5-8250U @ 1.60GHz

---

## Instalação e Execução

1. **Clonar o repositório:**

```bash
git clone https://github.com/franciscorrodrigues104/Projeto_detecao_objetos_SONAE_ARAUCO.git
