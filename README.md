# 🚜 Projeto de Deteção e Contagem de Tratores - Sonae Arauco

Este projeto utiliza visão computacional e **Deep Learning** para automatizar a deteção e monitorização de tratores em ambiente industrial. Desenvolvido durante o estágio na **Sonae Arauco**, o sistema otimiza o controlo logístico através da análise de vídeo em tempo real.

## 📋 Funcionalidades
* **Deteção em tempo real:** Identificação precisa de tratores em fluxos de vídeo CCTV.
* **Modelo Finalizado:** YOLOv8 treinado especificamente para as condições de iluminação e ambiente da fábrica.
* **Interface Web:** Dashboard em Flask para visualização das deteções e contagem automática.
* **Integração Cloud:** Armazenamento de dados no **Supabase** para relatórios e análise histórica.

---

## 🏗️ Estrutura do Repositório

O projeto seguiu uma metodologia rigorosa dividida em 5 etapas:

1.  **`passo_1_extracao_imagens_frames`**: Processamento de vídeos brutos para criação do dataset.
2.  **`passo_2_divisao_teste_treino`**: Segmentação de dados em treino e validação (Padrão YOLO).
3.  **`passo_3_treino_modelo`**: Configuração e execução do treino (100 épocas).
4.  **`passo_4_teste_modelo`**: Scripts de inferência e validação de performance.
5.  **`passo_5_detecao_front_end`**: Solução final com interface de utilizador e base de dados.

---

## 🚀 Tecnologias Utilizadas

* **IA:** [YOLOv8](https://ultralytics.com/) (Ultralytics)
* **Linguagem:** Python 3.10
* **Framework Web:** Flask
* **Base de Dados:** Supabase (PostgreSQL)
* **Bibliotecas:** OpenCV, PyTorch, PyInstaller

---

## 📊 Performance e Resultados do Treino

O treino do modelo foi **concluído com sucesso** (100 épocas), apresentando métricas sólidas para implementação:

* **Convergência:** O modelo atingiu estabilidade após a época 50, com a `cls_loss` (perda de classificação) consolidada abaixo de **0.9**, indicando alta fiabilidade na identificação.
* **Precisão:** Foram registados picos de precisão superiores a **70%** em ambientes de teste controlados.
* **Localização (Box Loss):** As Bounding Boxes demonstraram um ajuste preciso aos limites dos veículos, minimizando falsos positivos.

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
3.  **Executar a aplicação final (cmd):**
    ```bash
    python passo_5_detecao_front_end/app_detecao/codigo_trator_camara.py
    ```

---

**Autor:** Francisco Rodrigues  
**Instituição:** Sonae Arauco (Estágio Curricular)
