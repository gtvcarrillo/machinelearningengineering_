# Projeto ML – Predição de Aptidão à Maratona, Esforço (FC) e Semanas de Treino

Este projeto classifica atletas quanto à **aptidão** para completar uma maratona dentro de um **corte por gênero**
(**3h** para **homens** / **3h45** para **mulheres**), estima o **esforço fisiológico** necessário com base na
**frequência cardíaca (FC)** e prevê o **número de semanas de treino** para alcançar o alvo.

Implementação principal: **`marathon_end2end.py`** (executável no terminal/Jupyter).
Há também uma organização sugerida em módulos para uso no Jupyter (pacote `marathon/`).

---

## 🔍 Objetivos

1. **Apto (0/1)** – se o atleta já cumpre o corte por gênero.
2. **Esforço (Já atinge/Baixo/Moderado/Alto)** – quão exigente é alcançar o alvo combinando **gap de velocidade** e **FC necessária** versus **percentis pessoais** (P80/P90).
3. **Semanas até o alvo** – previsão do tempo de preparação considerando **evolução de velocidade** e **longão**.

---

## 📦 Requisitos

- Python 3.9+
- `pandas`, `numpy`, `scikit-learn`, `joblib`
- (opcional) `lightgbm` para melhor performance do classificador

Instalação rápida:
```bash
pip install -U pandas numpy scikit-learn joblib
# opcional
pip install lightgbm
```

---

## ▶️ Como executar

Com caminhos padrão já embutidos (Windows):
```bash
python marathon_end2end.py --train_model
```

Para especificar caminhos:
```bash
python marathon_end2end.py   --raw_path "C:/Users/SEU_USUARIO/Documents/raw-data-kaggle.csv"   --out_dir  "C:/Users/SEU_USUARIO/Documents/output"   --train_model
```
> No Jupyter, use `!python ...`. O script ignora argumentos desconhecidos do kernel (ex.: `-f kernel.json`).

---

## 🛠️ Pipeline – Etapas

### v1) Processamento dos Dados
- Limpeza, tipagem e criação de derivados por treino:
  - `km = distance_m/1000`, `speed_mps = distance_m/elapsed_s`, `pace_s_per_km = elapsed_s/km`
  - Filtros de plausibilidade: `km > 0.5` e `pace_s_per_km ∈ [150s, 1200s]`
- **Previsão de maratona (Riegel)** por treino:
  \\( \hat{T}_{42k} = T_{treino}\cdot(\tfrac{42.195}{dist_{treino}})^{1.06} \\)
- Guardamos por atleta a **melhor previsão** (menor tempo) entre treinos **≥10 km**: `best_pred_marathon_s`.

### 2) Agregação por atleta
- Volume/ritmo: `num_runs`, `total_km`, `longest_run_km`, `median_pace_s_per_km`, `best_pace_s_per_km`
- Terreno: `mean_elev_gain_per_km`
- Cardio: `avg_hr_bpm`
- Alvo por gênero: `target_s_gender = 10800s (M)` ou `13500s (F)`
- Velocidades: `v_pred_mps = 42195/best_pred_marathon_s`, `v_needed_mps = 42195/target_s_gender`
- **Apto**: `apto_genero = 1` se `best_pred_marathon_s ≤ target_s_gender`, senão 0
- Gap de performance: `esforco_extra_pct_genero = ((v_needed - v_pred)/v_pred)×100`

### 3) Relação FC–Velocidade e Esforço Fisiológico
- Para **cada atleta**, ajustamos **HR = a + b·velocidade** com treinos 8–25 km → `a_int`, `b_slope`
- Percentis pessoais de HR: `hr_p50`, `hr_p80`, `hr_p90`
- **FC necessária no alvo**: `hr_needed_bpm = a_int + b_slope·v_needed_mps`
- **Faixa de esforço (com HR) – `esforco_genero_hr`**:
  - **Já atinge**: `apto_genero = 1`
  - **Baixo**: gap < 5% **e** `hr_needed_bpm ≤ hr_p80`
  - **Moderado**: 5–15% **ou** `hr_p80 < hr_needed_bpm ≤ hr_p90`
  - **Alto**: >15% **ou** `hr_needed_bpm > hr_p90`

### 4) Estimativa de Semanas até o Alvo
- **Inclinação de melhora** (m/s por semana): regressão `speed_mps ~ weeks_since_first` em 8–25 km → `slope_speed_mps_per_week` (limitada a [0.01, 0.10])
- **Velocidade**: `weeks_speed_goal = (v_needed - v_pred)_+ / slope`
- **Longão**: elevar `longest_run_km` até **30 km** com +2 km/sem → `weeks_long_run`
- Tomamos o **máximo** e aplicamos multiplicador por esforço:
  - `Já atinge: 0.0`, `Baixo: 1.0`, `Moderado: 1.2`, `Alto: 1.5`
- **`weeks_to_target_genero_est_hr`** é arredondado para cima e limitado a 24 semanas

### 5) (Opcional) Classificador de `apto_genero`
- **Features**: `gender`, `num_runs`, `total_km`, `longest_run_km`, `median_pace_s_per_km`, `best_pace_s_per_km`,
  `avg_hr_bpm`, `mean_elev_gain_per_km`, `share_runs_ge_15k`, `best_pred_marathon_s`
- **Pré-processamento**: One-Hot em `gender`
- **Split**: treino/val/teste = 60%/20%/20%
- **Seleção de variáveis**: `SelectFromModel` com LGBM/RF (`threshold="median"`)
- **Treino final**: LightGBM (se instalado) ou RandomForest
- **Ajuste de threshold**: varredura 0.1–0.9 maximizando **F1** na validação
- **Métricas & artefatos**: importância de variáveis; matrizes de confusão; AUC/Precision/Recall ao longo do tempo; curvas PR

---

## 🧠 Modelos utilizados

**Determinísticos / Regras**
- **Fórmula de Riegel**: projeção de tempo de maratona a partir de treinos; gera `pred_marathon_s_from_run` e, por atleta, `best_pred_marathon_s`.
- **Regras de esforço (`esforco_genero_hr`)**: categorização via **gap de velocidade** + **FC necessária** vs **P80/P90** pessoais.

**Regressões individuais**
- **HR ~ velocidade** (por atleta): regressão linear (`numpy.polyfit`) → `a_int`, `b_slope`; permite estimar `hr_needed_bpm`.
- **Velocidade ~ tempo** (semanas): regressão linear (8–25 km) → `slope_speed_mps_per_week`.

**Classificador de aptidão (opcional)**
- **LightGBM (GBDT)** ou **RandomForest**; OHE em `gender`; `SelectFromModel` para seleção; ajuste de **threshold** por F1.

---

## 📚 Dicionário de Variáveis — detalhado

| Nome | Tipo/Unid. | Criada onde | Definição / Cálculo | Interpretação |
|---|---|---|---|---|
| `km` | km | por treino | `distance_m/1000` | Distância do treino. |
| `speed_mps` | m/s | por treino | `distance_m/elapsed_s` | Velocidade média do treino. |
| `pace_s_per_km` | s/km | por treino | `elapsed_s/km` | Ritmo médio do treino. |
| `pred_marathon_s_from_run` | s | por treino | Riegel: `elapsed_s*(42195/distance_m)^1.06` | Projeção de maratona a partir do treino. |
| `best_pred_marathon_s` | s | por atleta | `min(pred_marathon_s_from_run)` em ≥10k | Melhor projeção individual. |
| `num_runs` | contagem | por atleta | Nº de treinos válidos | Volume (quantidade). |
| `total_km` | km | por atleta | Soma de `km` | Volume (quilometragem). |
| `longest_run_km` | km | por atleta | Máximo de `km` | Capacidade de longões. |
| `median_pace_s_per_km` | s/km | por atleta | Mediana do `pace_s_per_km` | Ritmo típico. |
| `best_pace_s_per_km` | s/km | por atleta | Mínimo do `pace_s_per_km` | Melhor ritmo já registrado. |
| `avg_hr_bpm` | bpm | por atleta | Média de `avg_hr_bpm` | Esforço cardíaco médio. |
| `mean_elev_gain_per_km` | m/km | por atleta | `sum(elev_gain_m)/total_km` | Dureza do terreno. |
| `share_runs_ge_15k` | fração | por atleta | Proporção de treinos `km≥15` | Consistência de longos. |
| `target_s_gender` | s | por atleta | 10800 (M) / 13500 (F) | Corte por gênero. |
| `v_pred_mps` | m/s | por atleta | `42195/best_pred_marathon_s` | Velocidade equivalente atual. |
| `v_needed_mps` | m/s | por atleta | `42195/target_s_gender` | Velocidade necessária. |
| `apto_genero` | 0/1 | por atleta | `best_pred_marathon_s ≤ target_s_gender` | Já cumpre o corte? |
| `esforco_extra_pct_genero` | % | por atleta | `((v_needed - v_pred)/v_pred)*100` | % de velocidade a ganhar. |
| `a_int`, `b_slope` | bpm; bpm/(m/s) | por atleta | Regressão **HR = a + b·vel** | Parâmetros pessoais HR~vel. |
| `hr_p50`, `hr_p80`, `hr_p90` | bpm | por atleta | Percentis pessoais de HR | Referências individuais. |
| `hr_needed_bpm` | bpm | por atleta | `a_int + b_slope*v_needed_mps` | FC estimada na velocidade-alvo. |
| `esforco_genero_hr` | cat | por atleta | Regras com gap e HR | Dureza fisiológica para o alvo. |
| `weeks_since_first` | sem | por treino | `(timestamp - primeiro_treino)/7` | Tempo relativo (slope). |
| `slope_speed_mps_per_week` | m/s/sem | por atleta | Regressão **vel ~ semanas** | Melhora semanal. |
| `delta_v_needed` | m/s | por atleta | `(v_needed - v_pred)_+` | Gap de velocidade restante. |
| `weeks_speed_goal` | sem | por atleta | `delta_v_needed/slope` | Semanas p/ fechar gap de velocidade. |
| `weeks_long_run` | sem | por atleta | `ceil((30 - longest_run_km)/2)_+` | Semanas p/ elevar longão a 30 km. |
| `mult_effort_hr` | fator | por atleta | Mapa de `esforco_genero_hr` | Ajuste por dureza fisiológica. |
| `weeks_to_target_genero_est_hr` | sem | por atleta | `ceil(max(weeks_speed_goal,weeks_long_run)*mult)` (cap 24) | Estimativa final de semanas. |
| `selected_feature_names` | lista | modelagem | Atributos usados no classificador final | Transparência. |
| `threshold` | [0,1] | modelagem | Limiar ótimo por F1 (validação) | Ponto operacional. |
| `val_metrics`, `test_metrics` | dict | modelagem | AUC, Accuracy, Precision, Recall, F1 | Qualidade e generalização. |

---

## 🧾 Artefatos gerados no `out_dir`

- **Tabelas por atleta**:  
  `athlete_features_gender.csv`, `athlete_features_gender_hr.csv`, `athlete_final_weeks_gender_hr.csv`
- **Resumo**: `summary.json`
- **Modelagem (se \\`--train_model\\`)**:
  - `feature_importance.csv`, `feature_importance.png`
  - `cm_validation.png`, `cm_test.png`
  - `auc_over_time_val.png`, `auc_over_time_test.png`
  - `pr_over_time_val.png`, `pr_over_time_test.png`
  - `precision_recall_summary.csv`
  - `precision_recall_curve_val.csv`, `precision_recall_curve_test.csv`
  - `model_apto_genero_bundle.pkl`

---

## ✅ O modelo está sendo útil e eficiente?

**Critérios práticos** (no **conjunto de teste**):
- **AUC** ≥ 0,80 (bom); 0,90+ (excelente)
- **F1** ≥ 0,70 (bom); 0,60–0,69 (aceitável)
- **Gap validação→teste** (AUC e F1) ≤ 0,05 (evita overfitting)
- Ajuste conforme estratégia:
  - Prioriza **não perder aptos** → **Recall** ≥ 0,80
  - Prioriza **evitar falsos aptos** → **Precision** ≥ 0,80

**Script de avaliação rápida**:
```python
import json, pandas as pd
from pathlib import Path
import joblib

OUT_DIR = Path(r"C:/Users/SEU_USUARIO/Documents/output")
pr = pd.read_csv(OUT_DIR / "precision_recall_summary.csv")
m_val = pr[pr["split"]=="validação"].iloc[0].to_dict()
m_tst = pr[pr["split"]=="teste"].iloc[0].to_dict()
try:
    bundle = joblib.load(OUT_DIR / "model_apto_genero_bundle.pkl")
    auc_val = bundle["val_metrics"]["roc_auc"]; auc_tst = bundle["test_metrics"]["roc_auc"]
except Exception:
    auc_val = auc_tst = None

precision_t, recall_t, f1_t = float(m_tst["precision"]), float(m_tst["recall"]), float(m_tst["f1"])
precision_v, recall_v, f1_v = float(m_val["precision"]), float(m_val["recall"]), float(m_val["f1"])

ok_auc = (auc_tst is not None) and (auc_tst >= 0.80)
ok_f1  = f1_t >= 0.70
ok_gap = True
if (auc_tst is not None) and (auc_val is not None):
    ok_gap = abs(auc_val - auc_tst) <= 0.05 and abs(f1_v - f1_t) <= 0.05

print("=== VAL ===", {k:round(v,3) for k,v in dict(AUC=auc_val, P=precision_v, R=recall_v, F1=f1_v).items()})
print("=== TST ===", {k:round(v,3) for k,v in dict(AUC=auc_tst, P=precision_t, R=recall_t, F1=f1_t).items()})
print("
Veredito:")
print("AUC_tst >= 0.80:", "OK" if ok_auc else "NÃO")
print("F1_tst >= 0.70:", "OK" if ok_f1 else "NÃO")
print("Gaps val→tst ≤ 0.05:", "OK" if ok_gap else "NÃO")
if all([ok_auc or auc_tst is None, ok_f1, ok_gap]):
    print("
>> O modelo está **útil e eficiente** pelos critérios definidos.")
else:
    print("
>> O modelo **precisa de ajustes** (revise limiar e features).")
```

---

## ⚠️ Limitações & próximos passos

- **Riegel** é um proxy populacional; confirmações em provas/referências são desejáveis.
- **FC sem idade** usa percentis pessoais (P80/P90); depende da qualidade dos treinos com HR.
- **Inclinação de melhora** assume tendência linear; ciclos/lesões/sazonalidade podem afetar.
- Expandir features, usar **cross-validation** e ajustar hiperparâmetros (LGBM) podem elevar AUC/F1.

---

## 📎 Créditos

- Estrutura inspirada em notebook de referência compartilhado pelo usuário.
- Implementação e documentação deste README geradas em colaboração assistiva.

