# 🌟 Passos Mágicos - Predictive Analytics Dashboard

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-111111?style=for-the-badge&logo=XGBoost&logoColor=white)

Este projeto foi desenvolvido para o **Datathon da Fase 5 do curso de Data Analytics (Postech - FIAP)**, em parceria com a **Associação Passos Mágicos**. 

O objetivo principal é fornecer uma ferramenta analítica e preditiva robusta para monitorar o desempenho dos alunos, identificar riscos de evasão ou baixo desempenho e fornecer insights prescritivos para a equipe pedagógica.

## 🚀 Funcionalidades Principais

### 1. 🔍 Filtros Dinâmicos e Busca por RA
Busca avançada por Identificador do Aluno (RA). Ao selecionar um aluno, todos os filtros da aplicação (Ano e Pedra) se adaptam automaticamente à trajetória histórica daquele estudante, permitindo uma análise focada e sem erros de dados inexistentes.

### 2. 🎓 Diagnóstico Pedagógico (Aba Acadêmico)
*   **Modo Coletivo:** Matriz de Desempenho (IEG x IDA) com quadrantes coloridos (Estrela, Esforçado, etc.) e boxplots comparativos.
*   **Modo Individual:** 
    *   Gráfico de Radar comparando o aluno com a média do seu grupo (Pedra).
    *   Linha do tempo de evolução de indicadores.
    *   Diagnóstico textual automatizado com pontos fortes e fracos.

### 3. 🧠 Fatores Psicossociais (Aba Psicossocial)
*   **Análise de Risco com IA:** Probabilidade de risco calculada em tempo real por um modelo XGBoost (Acurácia > 80%).
*   **Gauge de Risco:** Medidor visual intuitivo com status dinâmico.
*   **Adequação Idade-Série (IAN):** Monitoramento de defasagem escolar com cards de alerta coloridos.

### 4. 🧪 Laboratório EDA
Ambiente para análise exploratória flexível, permitindo cruzamentos dinâmicos entre quaisquer indicadores (Eixo X, Eixo Y, Cor), auxiliando na descoberta de padrões e correlações ocultas.

### 5. 🔮 Simulador Preditivo
Interface para simular cenários. Ao alterar os indicadores de um aluno, o modelo de IA recalcula instantaneamente a probabilidade de risco e sugere recomendações pedagógicas imediatas.

---

## 🛠️ Stack Tecnológica

*   **Linguagem:** Python 3.10
*   **Frontend:** Streamlit
*   **Visualização:** Plotly, Graph Objects
*   **Ciência de Dados:** Pandas, Numpy, Scikit-learn
*   **Machine Learning:** XGBoost (Modelo Final de Classificação)
*   **Infraestrutura:** Docker & Docker-compose

---

## 📦 Como Executar

### Pré-requisitos
*   Docker e Docker-compose instalados.

### Passo a Passo
1. Clone o repositório:
   ```bash
   git clone git@github.com:gabiircode/predictive-analytics-passos-magicos.git
   cd predictive-analytics-passos-magicos
   ```

2. Suba os containers:
   ```bash
   docker-compose up -d --build
   ```

3. Acesse o dashboard:
   Abra o navegador em `http://localhost:8501`.

---

## 📂 Estrutura do Projeto

```text
├── app/                        # Código fonte do dashboard (Streamlit)
│   ├── passos_streamlit.py     # Script principal da aplicação
│   ├── modelo_risco_...pkl     # Modelo de IA treinado
│   └── dados_...csv            # Dados limpos e padronizados
├── notebooks/                  # Documentação e desenvolvimento do pipeline
│   └── passos_magicos.ipynb    # Notebook com ETL, EDA e Treino do Modelo
├── data/                       # Arquivos de dados originais (Excel)
├── docs/                       # Dicionários de dados e requisitos do projeto
└── docker-compose.yml          # Orquestração do ambiente
```

---

## 🛡️ Licença
Projeto desenvolvido para fins educacionais e de impacto social.

**Associação Passos Mágicos** - *Transformando vidas através da educação.*