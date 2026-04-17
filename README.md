# 🚜 Projeto de Deteção e Contagem de Tratores - Sonae Arauco

Este projeto utiliza visão computacional e **Deep Learning** para automatizar a deteção e monitorização de tratores em ambiente industrial. Desenvolvido durante o estágio na **Sonae Arauco**, o sistema otimiza o controlo logístico através da análise de vídeo em tempo real com alta precisão.

## 📋 Funcionalidades
* **Deteção de Alta Precisão:** Identificação de tratores com métricas de sucesso superiores a 99%.
* **Modelo Finalizado:** YOLOv8 treinado especificamente para o ambiente fabril da Sonae Arauco.
* **Interface Web:** Dashboard em Flask para visualização das deteções e contagem automática.
* **Integração Cloud:** Registo de dados no **Supabase** para análise histórica e relatórios.

---

## 🏗️ Estrutura do Repositório

O projeto seguiu uma metodologia estruturada em 5 etapas:

1.  **`passo_1_extracao_imagens_frames`**: Processamento de vídeos CCTV para criação do dataset.
2.  **`passo_2_divisao_teste_treino`**: Segmentação de dados (Train/Val) seguindo o padrão YOLO.
3.  **`passo_3_treino_modelo`**: Execução do treino robusto ao longo de 100 épocas.
4.  **`passo_4_teste_modelo`**: Scripts de inferência e validação final de métricas.
5.  **`passo_5_detecao_front_end`**: Aplicação completa (Flask + Supabase).

---

## 🚀 Tecnologias Utilizadas

* **IA:** YOLOv8 (Ultralytics)
* **Linguagem:** Python 3.12
* **Backend:** Flask & Supabase (PostgreSQL)
* **Hardware de Teste:** Intel Core i5-8250U @ 1.60GHz
* **Bibliotecas:** OpenCV, PyTorch (torch-2.10.0+cpu)

---

## 📊 Performance e Resultados Finais

O treino foi concluído com métricas de excelência após **100 épocas** (aprox. 3.7 horas de processamento):

* **mAP50 (Mean Average Precision):** **0.995 (99.5%)** - Indica uma precisão quase perfeita na deteção.
* **mAP50-95:** **0.99 (99%)** - Demonstra uma precisão extrema no ajuste das Bounding Boxes.
* **Precisão (P):** **0.974**
* **Recall (R):** **1.0** (O modelo detetou todos os casos de teste sem falhas).
* **Velocidade de Inferência:** ~66.3ms por imagem (ideal para monitorização em tempo real).

> O modelo demonstrou uma convergência sólida, com a perda de caixa (`box_loss`) a descer para **0.09** e a perda de classificação (`cls_loss`) para **0.11** na última época.

---

## 🛠️ Como Utilizar

1.  **Clonar o projeto (cmd):**
    ```bash
    git clone [https://github.com/franciscorrodrigues104/Projeto_detecao_objetos_SONAE_ARAUCO.git](https://github.com/franciscorrodrigues104/Projeto_detecao_objetos_SONAE_ARAUCO.git)
    ```
2.  **Instalar dependências (cmd):**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Executar a aplicação (cmd):**
    ```bash
    python passo_5_detecao_front_end/app_detecao/codigo_trator_camara.py
    ```

---

**Autor:** Francisco Rodrigues  
**Instituição:** Sonae Arauco (Estágio Curricular)
