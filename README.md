# Predição de Evasão Estudantil em Cursos de Tecnologia - UFOP

Este repositório contém a implementação prática da estratégia de predição de evasão estudantil desenvolvida como Trabalho de Conclusão de Curso (TCC) no Bacharelado em Ciência da Computação da Universidade Federal de Ouro Preto (UFOP).

## 📌 Sobre o Projeto
O objetivo principal é identificar precocemente a probabilidade de evasão de alunos em cursos da área de tecnologia utilizando registros acadêmicos e sociodemográficos. 

Através da comparação entre os modelos **KNN**, **XGBoost** e **Floresta Aleatória**, o algoritmo de **Floresta Aleatória** obteve o melhor desempenho (Acurácia de 89,00% e F1-Score de 87,32% para a classe de evasão), sendo selecionado para compor o pipeline operacional de predição.

## 📊 Atributos Utilizados
* Coeficiente de Rendimento (CR)
* Total de reprovações por nota
* Total de reprovações por falta
* Idade
* Sexo
* Modalidade de Ingresso (Cota / Ampla Concorrência)
* Naturalidade (Sede da instituição ou fora)

## 🛠️ Tecnologias e Bibliotecas
* **Python** (v3.10)
* **Pandas** & **NumPy** (Manipulação de dados)
* **Scikit-Learn** (Pré-processamento, pipeline e modelos)
* **XGBoost** (Gradient Boosting)

## 🚀 Como Executar

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/seu-usuario/predicao-evasao-ufop.git](https://github.com/seu-usuario/predicao-evasao-ufop.git)
   cd predicao-evasao-ufop
