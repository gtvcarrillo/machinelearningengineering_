# Predição de Aptidão para Maratona

## 1. Contexto do Projeto
Este projeto tem como objetivo **prever se um atleta será capaz de completar uma maratona dentro de limites de tempo de referência** usando dados históricos de corridas.  
O target foi definido de acordo com o **tempo previsto pelo modelo de Riegel**, considerando:

- **Homens:** ≤ 3h30  
- **Mulheres:** ≤ 4h00  

O foco é classificar cada atleta como **apto (1)** ou **não apto (0)**.

---

## 2. Metodologia
- Preparação de dados, agregação por atleta e cálculo do target com a fórmula de Riegel.
- Features selecionadas com `SelectKBest` e treinamento de modelos RandomForest e SVM.
- Avaliação com Precision, Recall, F1-Score, AUC e Matrizes de Confusão.

---

## 3. Estrutura do Pipeline
1. **Leitura dos Dados** – CSV de corridas individuais.
2. **Processamento dos Dados** – cálculo de pace, velocidade, agregação por atleta.
3. **Separação de Treino e Teste** – split estratificado de 80/20.
4. **Seleção de Variáveis** – `SelectKBest` para selecionar até 12 features.
5. **Treinamento de Modelos** – RandomForest e SVM.
6. **Importância de Variáveis** – análise para RandomForest.
7. **Avaliação de Resultados** – métricas e gráficos.

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
