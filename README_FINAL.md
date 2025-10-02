# Predição de Aptidão para Maratona

## 1. Contexto do Projeto
Este projeto tem como objetivo **prever se um atleta será capaz de completar uma maratona dentro de limites de tempo de referência** usando dados históricos de corridas.  
O target foi definido de acordo com o **tempo previsto pelo modelo de Riegel**, considerando:

- **Homens:** ≤ 3h30  
- **Mulheres:** ≤ 4h00  

> Observação: Os tempos de 3h30 para homens e 4h00 para mulheres foram retirados do tempo de qualificação da **Maratona de Boston**, utilizada como referência de aptidão.

O foco é classificar cada atleta como **apto (1)** ou **não apto (0)**.

---

## 2. Metodologia

### 2.1 Preparação dos Dados
- Dados importados de arquivos CSV contendo informações de corridas individuais:
  - Distância percorrida
  - Tempo de conclusão
  - Elevação
  - Frequência cardíaca média
  - Data da corrida
- Foram calculadas variáveis adicionais:
  - `dist_km`: distância em km
  - `dur_h`: duração em horas
  - `pace_min_km`: pace médio por km
  - `speed_kmh`: velocidade média
- Agregações por atleta:
  - Estatísticas descritivas (média, mediana, máximo, desvio padrão) de distância, pace, velocidade, frequência cardíaca e elevação
  - Total de quilômetros corridos
  - Volume semanal médio (`weekly_km`)

### 2.2 Cálculo do Target
O **tempo previsto de maratona** (`pred_marathon_h`) foi calculado a partir do **melhor desempenho em 10k** usando a fórmula de **Riegel**:

\[
t_2 = t_1 \times \left(\frac{d_2}{d_1}\right)^r
\]

- `t1` = tempo da prova conhecida (em horas)  
- `d1` = distância da prova conhecida (em km)  
- `t2` = tempo previsto para a distância alvo (maratona, 42,195 km)  
- `d2` = distância alvo (maratona, 42,195 km)  
- `r` = expoente de Riegel, geralmente 1.06  

> A fórmula assume que a performance diminui proporcionalmente à distância, considerando resistência e fadiga.

- **Target binário**:
  - `1` = atleta apto (predição ≤ tempo referência)  
  - `0` = atleta não apto

- **Classificação pelo Riegel**:
  - Número de atletas aptos segundo a fórmula: **XX** (substituir pelo valor calculado)
  - Número de atletas não aptos: **YY** (substituir pelo valor calculado)

---

## 3. Estrutura do Pipeline
1. **Leitura dos Dados** – carregamento do CSV e padronização de nomes de colunas.  
2. **Processamento dos Dados** – cálculo de pace, velocidade e agregação por atleta.  
3. **Separação de Treino e Teste** – split estratificado de 80/20.  
4. **Seleção de Variáveis** – `SelectKBest` com `mutual_info_classif`, selecionando até 12 features.  
5. **Treinamento de Modelos** – RandomForest e SVM.  
6. **Importância de Variáveis** – análise para RandomForest.  
7. **Avaliação de Resultados** – Precision, Recall, F1-Score, AUC, Matrizes de Confusão e gráficos.

---

## 4. Features Utilizadas
- `gender_M` – indicador de sexo masculino  
- `n_runs`, `total_km`, `avg_km`, `std_km`, `max_km`  
- `med_pace_min_km`, `avg_pace_min_km`, `std_pace_min_km`  
- `avg_speed_kmh`, `std_speed_kmh`  
- `avg_hr`, `std_hr`  
- `elev_gain_total`, `elev_gain_avg`, `elev_gain_std`  
- `weekly_km`  

---

## 5. Modelos Utilizados
- **Random Forest**
  - 400 árvores, Random state 42, N_jobs=-1
- **SVM Linear**
  - Kernel linear, probabilidade habilitada, Random state 42

---

## 6. Comparação de Resultados

### 6.1 Métricas de Avaliação
| Modelo         | Precision | Recall | F1-Score | AUC  |
|----------------|-----------|--------|----------|------|
| RandomForest   | 0.941     | 0.941  | 0.941    | 0.97 |
| SVM            | 0.938     | 0.882  | 0.909    | 0.95 |

### 6.2 Curvas e Gráficos
- **Curva ROC** dos modelos:  
![ROC Curve](roc_curve.png)
- **Precision e Recall**:  
![Precision/Recall](precision_recall.png)

---

## 7. Saídas do Pipeline
- **Modelos treinados**: `model_rf.pkl`, `model_svm.pkl`  
- **Métricas comparativas**: `metrics_comparison_rf_svm.csv`  
- **Matrizes de Confusão**: `confusion_matrices_rf_svm.csv`  
- **Gráficos**: `roc_curve.png`, `precision_recall.png`  
- **Tabela de comparação de Precision e Recall**: `precision_recall_comparison.csv`  

Todos os arquivos são exportados para a pasta de saída configurada (`OUTPUT_DIR`).

---

## 8. Como Executar
1. Instalar dependências:

```bash
pip install scikit-learn matplotlib joblib pandas
