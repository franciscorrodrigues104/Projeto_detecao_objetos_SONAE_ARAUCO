# Portfolio de Estágio Curricular - Sonae Arauco

**Autor:** Francisco Ribeiro Rodrigues  
**Instituição:** Sonae Arauco (Estágio Curricular)

Este repositório reúne os projetos de Visão Computacional e Inteligência Artificial desenvolvidos durante o estágio curricular na Sonae Arauco. O foco principal dos trabalhos consistiu na automatização de processos industriais através da utilização de modelos YOLO, processamento de imagem e dashboards de monitorização em tempo real.

---

# Índice

1. #projeto-1-deteção-e-contagem-de-tratores
2. [Projeto 2: Deteção de Falhas em Lotes](#[Projeto 3: Deteção e Análise de Defeitos em Manta Contínua](#projeto-3-deteção-e-análise-destalação-e-execução

---

# Projeto 1: Deteção e Contagem de Tratores

Sistema desenvolvido para automatizar a monitorização logística de tratores em ambiente industrial, permitindo contabilizar automaticamente os veículos que circulam numa determinada zona da fábrica.

## Funcionalidades

- **Deteção de Alta Precisão:** Identificação automática de tratores com métricas superiores a 99%.
- **Modelo Personalizado:** YOLOv8 treinado especificamente com imagens reais da Sonae Arauco.
- **Dashboard Web:** Interface desenvolvida em Flask para monitorização em tempo real.
- **Integração Cloud:** Armazenamento de dados históricos em Supabase para análise posterior.

## Performance e Resultados

Treino realizado durante **100 épocas** (aproximadamente 3.7 horas).

| Métrica | Resultado |
|----------|------------|
| mAP50 | 99.5% |
| mAP50-95 | 99.0% |
| Precisão (P) | 97.4% |
| Recall (R) | 100% |
| Tempo de Inferência | ~66.3 ms |

## Principais Resultados

- Automatização completa da contagem de tratores.
- Eliminação da necessidade de registo manual.
- Monitorização contínua da movimentação logística.

---

# Projeto 2: Deteção de Falhas em Lotes

Sistema de visão computacional desenvolvido para detetar automaticamente falhas e desalinhamentos em lotes de produção.

## Funcionalidades

- **Monitorização em Tempo Real**
- **Deteção Automática de Falhas**
- **Cooldown Inteligente de 6 segundos**, evitando múltiplos registos da mesma ocorrência.
- **Dashboard Dinâmico** para consulta de alertas e histórico.

## Performance e Resultados

### Modelo de Deteção de Falhas em Lotes

| Métrica | Resultado |
|----------|------------|
| mAP50 | 53.4% |
| mAP50-95 | 18.8% |
| Precisão (P) | 76.3% |
| Recall (R) | 50.9% |
| Tempo de Inferência | ~5.8 ms |

## Principais Resultados

- Identificação automática de desalinhamentos.
- Redução da necessidade de inspeção manual.
- Validação da viabilidade de utilização de IA em cenários com elevada variabilidade visual.

---

# Projeto 3: Deteção e Análise de Defeitos em Manta Contínua

Sistema avançado de inspeção visual industrial desenvolvido para monitorizar automaticamente a qualidade de mantas em linha de produção.

Este projeto combina **segmentação de imagem, deteção de defeitos, tracking de objetos, armazenamento histórico e análise estatística em tempo real**, constituindo o projeto tecnicamente mais completo desenvolvido durante o estágio.

## Funcionalidades

### Segmentação da Manta

- Segmentação automática da área útil da manta através de YOLO Segmentation.
- Eliminação do fundo e das zonas irrelevantes da imagem.
- Identificação da região de inspeção em tempo real.

### Deteção de Defeitos

- Identificação automática de defeitos na superfície da manta.
- Diferenciação entre múltiplas classes de defeitos.
- Configuração de níveis de confiança por classe.

### Tracking Inteligente

- Integração com **ByteTrack**.
- Evita contagens duplicadas.
- Acompanha o mesmo defeito ao longo da sequência de vídeo.

### Dashboard Industrial

- Visualização do vídeo processado em tempo real.
- Atualização contínua das métricas operacionais.
- Monitorização do sistema através de browser.

### Análise Estatística

- Histograma por tipo de defeito.
- Heatmap horizontal da manta.
- Tendência horária dos defeitos.
- Ranking dos defeitos mais frequentes.

### Armazenamento Automático

- Registo histórico em ficheiros CSV organizados por dia.
- Armazenamento automático das imagens dos defeitos detetados.
- Histórico de inspeções para análise futura.

### Robustez Operacional

- Reconexão automática à câmara.
- Arquitetura multi-thread para elevada fluidez.
- Monitorização contínua de FPS e latência.

---

## Arquitetura da Solução

O sistema foi desenvolvido com uma arquitetura otimizada para operação contínua em ambiente industrial.

### Thread de Captura

Responsável pela aquisição contínua do stream MJPEG da linha de produção.

### Thread de Inferência

Executa:

- Segmentação da manta
- Deteção de defeitos
- Tracking dos defeitos

### Thread de Escrita

Responsável pelo armazenamento assíncrono de:

- Imagens
- Registos históricos
- Estatísticas

### Thread de Monitorização

Monitoriza:

- FPS da câmara
- FPS de inferência
- Latência média
- Estado do sistema
- Utilização da fila de gravação

---

## Performance e Resultados

### Modelo de Segmentação da Manta

Treinado durante **100 épocas**.

| Métrica | Resultado |
|----------|------------|
| Precision (B) | ~99.8% |
| Recall (B) | ~99.8% |
| mAP50 (B) | ~99.5% |
| mAP50-95 (B) | ~99.0% |
| Precision (M) | ~99.8% |
| Recall (M) | ~99.8% |
| mAP50 (M) | ~99.5% |
| mAP50-95 (M) | ~99.5% |

### Modelo de Deteção de Defeitos

Treinado durante **100 épocas**.

| Métrica | Resultado |
|----------|------------|
| Precisão (P) | ~97.0% |
| Recall (R) | ~88.5% |
| mAP50 | ~93.0% |
| mAP50-95 | ~69.5% |

## Principais Resultados

- Inspeção automática da manta em tempo real.
- Segmentação da manta com desempenho superior a 99%.
- Deteção automática de defeitos com mAP50 superior a 93%.
- Eliminação de contagens duplicadas através de tracking.
- Criação de histórico de defeitos para análise de qualidade.
- Desenvolvimento de dashboard industrial em tempo real.
- Identificação de padrões de defeitos através de heatmaps.
- Sistema preparado para operação contínua em ambiente produtivo.

---

# Tecnologias Utilizadas

- **Deep Learning:** YOLOv8, YOLO11s, PyTorch
- **Visão Computacional:** OpenCV
- **Tracking de Objetos:** ByteTrack
- **Processamento Numérico:** NumPy
- **Backend:** Flask
- **Base de Dados:** Supabase (PostgreSQL)
- **Frontend:** HTML, JavaScript, HTMX
- **Linguagem:** Python 3.12

## Hardware de Teste

- Intel Core i5-8250U @ 1.60 GHz

---

# Instalação e Execução

## Clonar o repositório

```bash
git clone https://github.com/franciscorrodrigues104/Projeto_detecao_objetos_SONAE_ARAUCO.git
cd Projeto_detecao_objetos_SONAE_ARAUCO
```

## Instalar dependências

```bash
pip install -r requirements.txt
```

## Executar aplicação

```bash
python app.py
```

---

# Conclusão

Ao longo do estágio curricular foi possível desenvolver e validar soluções de Visão Computacional aplicadas a diferentes cenários industriais, desde a monitorização logística até à inspeção automática da qualidade da produção.

Os projetos demonstram a aplicação prática de modelos YOLO em ambiente industrial, integrando deteção de objetos, segmentação, tracking, dashboards web e armazenamento histórico de dados para apoio à tomada de decisão operacional.

O **Projeto de Deteção e Análise de Defeitos em Manta Contínua** representou a solução mais completa desenvolvida durante o estágio, combinando Inteligência Artificial, Visão Computacional e Engenharia de Software numa aplicação industrial de monitorização em tempo real.