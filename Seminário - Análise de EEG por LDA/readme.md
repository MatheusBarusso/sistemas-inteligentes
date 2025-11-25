# Projeto Análise de EEG por LDA

Este repositório contém os scripts necessários para reproduzir o treinamento e análise do modelo de classificação.

## Pré-requisitos

Para executar este projeto, você precisará do seguinte ambiente configurado:

* **MATLAB** (Versão recomendada: R2020b ou superior)
* **Statistics and Machine Learning Toolbox** (Essencial para as funções de classificação e treinamento)

## Instalação e Configuração dos Dados

Como o conjunto de dados é grande/externo, ele não está incluído diretamente neste repositório. Siga os passos abaixo para configurar o ambiente corretamente:

1.  Clone ou baixe este repositório em sua máquina local.
2.  **Baixe a pasta do dataset clicando no link**:
    *   https://drive.google.com/drive/folders/1fZoN_WQQLXgxzkbSUNqHG3MrqWLXf7re?usp=sharing
3.  **Organize os arquivos**:
    * Extraia/coloque a pasta chamada `dataset` no **mesmo diretório** onde se encontra o arquivo `Main.m`.

A estrutura de pastas deve ficar exatamente assim:

```text
sistemas-inteligentes/
│
├── Seminário - Análise de EEG por LDA/   <-- Abra esta pasta
│   ├── dataset/                          <-- A pasta baixada deve ficar AQUI
│   ├── Main.m                            <-- Script principal
│   ├── heavy_computing/                       
│   └── ...
│
├── Trabalho 1 - Redes Neurais/
├── Trabalho 2 - Busca e Otimização/
├── Trabalho 3 - Busca Discreta/
├── Trabalho 4 - Aprendizado Supervisionado/
├── Trabalho 4 - Aprendizado Supervisionado/
├── .gitignore
└── README.md