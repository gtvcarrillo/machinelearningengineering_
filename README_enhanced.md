# Projeto ML – Maratona: Aptidão, Esforço (FC) e Semanas de Treino

Este repositório traz um pipeline que:
1) **Classifica aptidão** para completar a maratona em **3h (M)** / **3h45 (F)**;  
2) **Estima esforço fisiológico** com base em **frequência cardíaca (FC)**;  
3) **Prevê semanas de treino** necessárias para atingir o alvo.

---

## 🧠 Modelos Utilizados (com clareza)

### 1) Componentes determinísticos (sem treino)
- **Fórmula de Riegel**: projeta tempo de maratona a partir de um treino. É uma relação empírica:
  \[
    \hat{T}_{42k} = T_{treino} \cdot \left(\frac{42.195}{dist_{treino}}\right)^{1.06}
  \]
  Usamos o **menor** tempo projetado entre treinos ≥10 km (por atleta): `best_pred_marathon_s`.

- **Regras de esforço (`esforco_genero_hr`)**: combinam  
  (a) o **gap de velocidade** até o alvo por gênero e  
  (b) a **FC necessária** para correr nessa velocidade, comparada a **percentis pessoais** (P80/P90) de FC.  
  Resultado categórico: **Já atinge / Baixo / Moderado / Alto**.

### 2) Regressões individuais (por atleta)
- **HR ~ velocidade** (linear): estima `a_int` e `b_slope` para prever a FC na velocidade-alvo (`hr_needed_bpm`).
- **Velocidade ~ tempo** (semanas) (linear): estima a **inclinação de melhora** `slope_speed_mps_per_week` (m/s por semana).

> Essas regressões não “comparam” atletas entre si: são **curvas pessoais** para traduzir histórico em esforço e ritmo de evolução.

### 3) Classificador de aptidão (opcional, supervisionado)
- **Algoritmo principal**: **LightGBM** (Gradient Boosted Decision Trees);  
  **Fallback**: **RandomForest**, caso LightGBM não esteja instalado.
- **Pré-processamento**: One-Hot Encoding apenas em `gender`.
- **Seleção de variáveis**: `SelectFromModel` com **limiar pela mediana** das importâncias de um modelo-base.
- **Ajuste operacional**: busca de **threshold** no intervalo [0.1, 0.9] para **maximizar F1** na *validação* e aplicação desse threshold no *teste*.
- **Métricas**: ROC AUC, Accuracy, Precision, Recall, F1; matrizes de confusão; AUC/Precision/Recall **ao longo do tempo** (robustez temporal).

---

## 🔢 Métricas do Modelo & Interpretação

> Substitua os valores abaixo pelos **seus resultados** (gerados no `output/`):
> - `precision_recall_summary.csv` (Precision, Recall, F1 por split)
> - `model_apto_genero_bundle.pkl` (contém AUC de validação e teste)

### Resumo Numérico (preencha)
| Split      | Precision | Recall | F1  | AUC  | Threshold |
|------------|-----------|--------|-----|------|-----------|
| Validação  | 0.XX      | 0.XX   | 0.XX| 0.XX | 0.XX      |
| Teste      | 0.XX      | 0.XX   | 0.XX| 0.XX | 0.XX      |

**Como interpretar:**
- **Precision**: dos identificados como aptos, quantos realmente são. Alta precisão → poucos falsos aptos.
- **Recall**: dos realmente aptos, quantos o modelo identifica. Alto recall → poucos aptos “perdidos”.
- **F1**: equilíbrio entre Precision e Recall (bom para cenários sem preferência clara).
- **ROC AUC**: capacidade do score em ordenar aptos vs não aptos, **independente do threshold**.
- **Threshold**: ponto operacional escolhido (otimizado por F1).  
  Ajuste conforme sua política:
  - Foco em **não perder aptos** → **reduza** o threshold (↑ Recall, ↓ Precision);
  - Foco em **evitar falsos aptos** → **aumente** o threshold (↑ Precision, ↓ Recall).

### Gaps de generalização (val → teste)
- Idealmente **≤ 0,05** em **AUC** e **F1**.  
  Gaps maiores sugerem **overfitting** ou **mudança de distribuição** (drift).

### Curvas e Gráficos (o que observar)
- **Matriz de Confusão (Val/Teste)**: padrão de erros (FP vs FN).
- **AUC ao longo do tempo**: estabilidade temporal; quedas podem indicar drift.
- **Precision & Recall ao longo do tempo**: comportamento operacional em janelas de tempo.

> Registre em 1–2 parágrafos o “veredito”:  
> - Se AUC_test ≥ 0.80 e F1_test ≥ 0.70, e gaps pequenos → **modelo eficiente**.  
> - Caso contrário → ajustar threshold e/ou enriquecer features.

---

## 🔍 Interpretação das Variáveis

### Direção de impacto (intuitiva)
- **Capacidade atual**
  - `best_pred_marathon_s` (s): **menor** (melhor) → mais apto.
  - `best_pace_s_per_km` / `median_pace_s_per_km` (s/km): **menor** (ritmo mais rápido) → mais apto.
- **Volume**
  - `total_km`, `num_runs`: **maiores** costumam indicar base melhor → maior probabilidade de aptidão.
  - `longest_run_km`: **maior** sugere capacidade para longões → favorece aptidão.
- **Terreno**
  - `mean_elev_gain_per_km`: **maior** indica treinos mais duros; pode sinalizar boa força/ resistência, mas depende do contexto.
- **Cardio**
  - `avg_hr_bpm`: valores muito altos crônicos podem indicar treinos sempre duros; interpretação depende do atleta (por isso usamos percentis pessoais).
- **Estrutura de treinos**
  - `share_runs_ge_15k`: **maior** indica consistência de longos → favorece aptidão.

### Variáveis de esforço (com FC)
- `a_int`, `b_slope`: descrevem a **curva pessoal** HR~velocidade.
- `hr_p80`, `hr_p90`: **marcos pessoais** de esforço.
- `hr_needed_bpm`: FC estimada para correr na **velocidade-alvo**.
- `esforco_genero_hr`:  
  - **Já atinge**: sem esforço adicional;  
  - **Baixo**: gap<5% e HR≤P80 (ajustes finos);  
  - **Moderado**: 5–15% **ou** HR entre P80–P90;  
  - **Alto**: >15% **ou** HR>P90 (exigência alta).

### Semanas de treino
- `slope_speed_mps_per_week`: melhora semanal esperada (m/s/sem).
- `weeks_speed_goal`: semanas para fechar **gap de velocidade**.
- `weeks_long_run`: semanas para elevar **longão** até 30 km (+2 km/sem).
- `weeks_to_target_genero_est_hr`: máximo entre as duas, **ajustado** pela dureza de esforço (multiplicador), **capado** em 24 semanas.

---

## 🧾 Artefatos gerados (pasta `output/`)

- **Tabelas por atleta**:  
  `athlete_features_gender.csv`, `athlete_features_gender_hr.csv`, `athlete_final_weeks_gender_hr.csv`
- **Resumo**: `summary.json`
- **Modelagem (se `--train_model`)**:
  - `feature_importance.csv`, `feature_importance.png`
  - `cm_validation.png`, `cm_test.png`
  - `auc_over_time_val.png`, `auc_over_time_test.png`
  - `pr_over_time_val.png`, `pr_over_time_test.png`
  - `precision_recall_summary.csv`
  - `precision_recall_curve_val.csv`, `precision_recall_curve_test.csv`
  - `model_apto_genero_bundle.pkl`

---

## 🧪 Como preencher automaticamente a seção de métricas

No Jupyter, rode:

```python
import json, pandas as pd, joblib
from pathlib import Path

OUT_DIR = Path(r"C:/Users/SEU_USUARIO/Documents/output")

pr = pd.read_csv(OUT_DIR / "precision_recall_summary.csv")
val = pr[pr["split"]=="validação"].iloc[0]
tst = pr[pr["split"]=="teste"].iloc[0]

try:
    bundle = joblib.load(OUT_DIR / "model_apto_genero_bundle.pkl")
    auc_val = bundle["val_metrics"]["roc_auc"]
    auc_tst = bundle["test_metrics"]["roc_auc"]
except Exception:
    auc_val = auc_tst = float("nan")

print("Tabela para colar no README:")
print(f"| Validação | {val['precision']:.3f} | {val['recall']:.3f} | {val['f1']:.3f} | {auc_val:.3f} | {val['threshold']:.2f} |")
print(f"| Teste     | {tst['precision']:.3f} | {tst['recall']:.3f} | {tst['f1']:.3f} | {auc_tst:.3f} | {tst['threshold']:.2f} |")
```

---

## ⚠️ Limitações e Próximos Passos

- **Riegel** é proxy populacional; validação com provas/referências é desejável.
- **FC sem idade** usa percentis pessoais (P80/P90); qualidade de HR influencia.
- **Slope linear** simplifica evolução; ciclos/lesões podem alterar.
- Melhorias: enriquecer features (consistência semanal, % tempo HR>P80, variação de ritmo, ganho normalizado), **cross-validation** e **HPO** (LightGBM).

---

## 📎 Créditos

- Estrutura inspirada em notebook de referência compartilhado.
- Implementação e documentação deste README em colaboração assistiva.
