Boa! Finalmente conseguiste "domar" o Git. Esse erro dos ficheiros grandes é o batismo de fogo de quase todos os programadores.

Agora que o teu repositório está limpo e funcional, aqui tens um README.md profissional e bem estruturado. Podes copiar o código abaixo, colar no teu ficheiro README.md (substituindo o que lá estiver) e fazer o upload final.

Markdown
# 🚜 Projeto de Deteção e Contagem de Tratores - Sonae Arauco

Este projeto utiliza visão computacional e **Deep Learning** para automatizar a deteção e monitorização de tratores em ambiente industrial. Desenvolvido durante o estágio na **Sonae Arauco**, o sistema visa otimizar o controlo logístico e operacional.

## 📋 Funcionalidades
* **Deteção em tempo real:** Identificação de tratores através de fluxos de vídeo (CCTV).
* **Treino Personalizado:** Modelo YOLOv8 treinado com dataset específico do ambiente de fábrica.
* **Interface Web:** Front-end para visualização das deteções e contagem.
* **Integração com Cloud:** Registo de dados no **Supabase** para análise histórica.

---

## 🏗️ Estrutura do Repositório

O projeto está organizado por etapas de desenvolvimento para garantir a reprodutibilidade:

1.  **`passo_1_extracao_imagens_frames`**: Scripts para converter vídeos brutos em frames para anotação.
2.  **`passo_2_divisao_teste_treino`**: Organização do dataset em conjuntos de treino e validação.
3.  **`passo_3_treino_modelo`**: Configuração e execução do treino do modelo YOLOv8.
4.  **`passo_4_teste_modelo`**: Validação da precisão do modelo em novos dados.
5.  **`passo_5_detecao_front_end`**: Aplicação completa com interface Flask e integração com base de dados.

---

## 🚀 Tecnologias Utilizadas

* **IA:** [YOLOv8](https://ultralytics.com/) (Ultralytics)
* **Linguagem:** Python 3.10
* **Bibliotecas:** OpenCV, PyTorch, Flask
* **Base de Dados:** Supabase
* **Ambiente:** Virtualenv (venv)

---

## 📊 Status do Treino (Atual)

O modelo encontra-se atualmente em fase de treino (100 épocas previstas).
* **Resultados preliminares:** O modelo demonstrou convergência rápida após a época 50, com a `cls_loss` (perda de classificação) a descer abaixo de **1.0**.
* **Precisão:** Picos de confiança superiores a **65%** em condições iniciais.

---

## 🛠️ Como Utilizar

1.  **Clonar o projeto:**
    ```bash
    git clone [https://github.com/franciscorrodrigues104/Projeto_detecao_objetos_SONAE_ARAUCO.git](https://github.com/franciscorrodrigues104/Projeto_detecao_objetos_SONAE_ARAUCO.git)
    ```
2.  **Instalar dependências:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Executar a deteção:**
    Navegar até a pasta do front-end e executar o script principal:
    ```bash
    python passo_5_detecao_front_end/app_detecao/codigo_trator_camara.py
    ```

---

**Autor:** Francisco Rodrigues  
**Instituição:** Sonae Arauco (Estágio Curricular)
