# Projeto ML – Maratona: Aptidão, Esforço (FC) e Semanas de Treino (Métricas Preenchidas)

## 🔢 Métricas do Modelo (extraídas do output)
| Split      | Precision | Recall | F1  | AUC  | Threshold |
|------------|-----------|--------|-----|------|-----------|
| Validação  | 0.688 | 1.000 | 0.815 | n/a | 0.18 |
| Teste      | 0.769 | 0.833 | 0.800 | n/a | 0.18 |

**Interpretação rápida**  
- **Precision**: dos previstos como aptos, quantos realmente são.  
- **Recall**: dos aptos reais, quantos o modelo captura.  
- **F1**: equilíbrio entre Precision e Recall.  
- **AUC (ROC)**: qualidade do score independentemente do threshold.

---

## 📊 Gráficos principais
![Importância de Variáveis](sandbox:/output_extracted/output/feature_importance.png)

![Matriz de Confusão – Validação](sandbox:/output_extracted/output/cm_validation.png)

![Matriz de Confusão – Teste](sandbox:/output_extracted/output/cm_test.png)

![AUC ao longo do tempo – Validação](sandbox:/output_extracted/output/auc_over_time_val.png)

![AUC ao longo do tempo – Teste](sandbox:/output_extracted/output/auc_over_time_test.png)

---

## 🧠 Modelos utilizados (resumo)
- **Riegel** (determinístico) para projeção de maratona por treino (melhor por atleta = `best_pred_marathon_s`).
- **Regras de esforço com FC** (gap de velocidade + `hr_needed_bpm` vs P80/P90 pessoais) gerando `esforco_genero_hr`.
- **Regressões individuais**: `HR ~ velocidade` (gera `a_int`, `b_slope`, `hr_needed_bpm`) e `velocidade ~ tempo` (gera `slope_speed_mps_per_week`).
- **Classificador (opcional)**: **LightGBM** (ou **RandomForest**), OHE em `gender`, seleção por importância (SelectFromModel), ajuste de **threshold** por F1.

---

## ✅ Veredito (regra de bolso)
Se **AUC_test ≥ 0.80** e **F1_test ≥ 0.70**, com gaps val→teste pequenos (≤ 0.05), o modelo está **eficiente**.  
Caso contrário, ajuste o threshold (trade-off Precision/Recall) e/ou enriqueça features.

(Gerado automaticamente a partir de `output.zip`)
