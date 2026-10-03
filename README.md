# Earthquake-Damage-Prediction
Machine learning project predicting earthquake building damage grades using EDA, preprocessing, multiple models, tuning, ordinal evaluation, and production-model selection.
# Earthquake Damage Prediction (PRCP-1015)

Predicting how badly a building was damaged in the 2015 Gorkha earthquake in Nepal, from its location, structure and use.

The target, `damage_grade`, is **ordinal**:

| grade | meaning |
|---|---|
| 1 | low damage |
| 2 | medium damage |
| 3 | almost complete destruction |

The whole project lives in one Jupyter notebook: data analysis, preprocessing, model comparison, tuning, a one-time test evaluation, interpretation, suggestions for seismologists, and a report on the challenges faced.

---

## Results at a glance

**Production model:** LightGBM with native categorical handling, tuned, plus a tuned decision rule on the predicted probabilities.

| | Macro F1 | Accuracy | QWK | Severe errors (1↔3) | Log loss |
|---|---|---|---|---|---|
| 5-fold cross-validation | 0.7082 | 0.7433 | 0.6333 | 0.58% | 0.5668 |
| **Held-out test set (52,121 buildings)** | **0.7171** | **0.7510** | **0.6462** | **0.52%** | **0.5512** |

- 95% bootstrap interval for test macro F1: **0.7125 to 0.7215**.
- Compared with a simpler, about 4× cheaper LightGBM, the production model is ahead by +0.0050 macro F1 on the test set (95% interval +0.0020 to +0.0080).
- **Ranking buildings by predicted risk:** inspecting the top 20% of buildings, ranked by predicted probability of grade 3, reaches **51%** of all destroyed buildings. The top 10% reaches 28%, and 95% of that short list really is grade 3.

---

## Dataset

- Source: [DrivenData, Richter's Predictor: Modeling Earthquake Damage](https://www.drivendata.org/competitions/57/nepal-earthquake/).
- `train_values.csv`: 260,601 buildings × 39 columns (`building_id` + 38 features).
- `train_labels.csv`: `building_id`, `damage_grade`.
- Features: 3 nested geographic ids (31 / 1,414 / 11,595 distinct values), 5 numeric columns (floors, age, area, height, families), 8 obfuscated categorical columns, and 22 binary flags for superstructure material and secondary use.
- Class balance: grade 1 = 9.6%, grade 2 = 56.9%, grade 3 = 33.5%.

The data files are not included in this repository. Download them and point `DATA_DIR` in the notebook (Section 2) to the folder that contains both CSVs.

---

## Approach

1. **Data checks and EDA.** I set aside a 20% stratified test set before looking at any feature-target relationship. All EDA uses the training part only.
2. **Evaluation framework.** The same 5 stratified folds for every model. All preprocessing sits inside pipelines, so it is refitted on each fold. Metrics chosen for an ordinal, imbalanced target: macro F1 (main), QWK, MAE, severe-error rate (1↔3 mistakes), per-grade recall, accuracy and log loss.
3. **Preprocessing driven by the EDA:**
   - The `age = 995` code is treated as "unknown": a separate flag, with the value replaced by the training-fold median.
   - High-cardinality geography uses cross-fitted target encoding, or LightGBM's native categorical handling.
   - Rare categories are grouped for one-hot encoding.
4. **Models compared:**
   - Baselines: majority class, random, and location only.
   - Logistic regression, with and without class weighting.
   - Random forest, LightGBM and XGBoost.
   - LightGBM with native categories.
   - Two ordinal approaches: regression with learned cut points, and cumulative binary models (Frank and Hall).
   - CatBoost was attempted but dropped because of a scikit-learn compatibility error.
5. **Decision rule.** Probabilities of grades 1 and 3 are re-weighted before choosing a grade, with the weights chosen on out-of-fold predictions only. Every probabilistic model gets the same treatment.
6. **Tuning.** `RandomizedSearchCV`, with the same budget, folds and objective (log loss) for each finalist.
7. **Test evaluation.** The production model is chosen on CV evidence first, and then evaluated on the test set exactly once, with bootstrap confidence intervals.
8. **Interpretation.** Grouped permutation importance and TreeSHAP contributions, plus a capture curve for inspection planning.

---

## Model comparison (cross-validation, tuned decision rule)

| Model | Macro F1 | QWK | Severe err | Fit / predict per fold |
|---|---|---|---|---|
| Logistic regression | 0.6796 | 0.606 | 0.50% | 6.0 s / 0.1 s |
| Regression + learned cut points (LightGBM) | 0.6986 | 0.629 | **0.33%** | 10.2 s / 0.3 s |
| Random forest | 0.7031 | 0.629 | 0.51% | 14.7 s / 0.5 s |
| LightGBM, per-grade target encoding | 0.7053 | 0.629 | 0.60% | **7.5 s / 0.8 s** |
| XGBoost, per-grade target encoding, tuned | 0.7064 | 0.631 | 0.58% | 44.6 s / 1.2 s |
| LightGBM, per-grade target encoding, tuned | 0.7070 | 0.630 | 0.56% | 41.9 s / 4.4 s |
| **LightGBM, native categories, tuned** | **0.7082** | **0.633** | 0.58% | 31.4 s / 3.4 s |

Main takeaways:

- **Tree models beat logistic regression** by about 0.025 macro F1. The best tree configurations, however, are within 0.003 of each other.
- **The decision rule helped more than hyperparameter tuning:** +0.0065 to +0.0163 macro F1, versus at most +0.0030 from tuning.
- **The production model is not best on every metric.** Regression with learned cut points makes the fewest severe errors, and the plain target-encoded LightGBM is the cheapest model in the top group.

---

## Key findings

- **Location is by far the strongest predictor.** Grade-3 share ranges from 8.8% to 80.6% across top-level regions, and shuffling the geography raises test log loss by 0.603, against 0.082 for the next feature group.
- **Foundation type and wall material show the clearest construction effects.** Mud-mortar stone and foundation type `r` are associated with more damage. Engineered reinforced concrete, cement-mortar brick and foundation type `i` are associated with less. These patterns hold within the same region.
- **Timber and bamboo only looked protective.** Their effect disappeared once location was controlled for.
- **Age matters up to about 25–30 years,** and then levels off.

All of these are **associations in observational data from one earthquake**, not proven causes. The categorical codes are obfuscated in the source data, so letters like `r` and `i` can't be mapped to real construction types here.

---

## Repository structure

```
├── Earthquake_Damage_Prediction___PRCP-1015.ipynb   # full project: EDA, models, reports
└── README.md
```

Running the notebook also creates `checkpoint_*.joblib` files (saved results) and `production_model.joblib` (the fitted model together with its decision weights).

---

## How to run

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn lightgbm xgboost joblib
jupyter notebook Earthquake_Damage_Prediction___PRCP-1015.ipynb
```

1. Set `DATA_DIR` in Section 2 to the folder containing `train_values.csv` and `train_labels.csv`.
2. Run the cells top to bottom.

**Library versions used:** pandas 3.0.2, NumPy 2.5.3, scikit-learn 1.9.1, LightGBM 4.7.0 and XGBoost 3.4.1. The notebook relies on `TargetEncoder`, so it needs scikit-learn 1.3 or newer.

**Run time:** the three hyperparameter searches took about 2.3 hours on a laptop (22, 32 and 82 minutes), so a full top-to-bottom run takes roughly 2.5 to 3 hours. Keep the machine awake during long cells, because a sleep in the middle of a search froze the kernel once (see the challenges section in the notebook).

---

## Limitations

- **One earthquake and one region.** Results may not transfer to events with a different epicentre or building stock.
- **No shaking-intensity or soil variables,** so the geography effect can't be split into ground motion and regional building practice.
- **Random split, not by region.** Almost all test buildings fall in areas seen during training, so performance on a completely new region would very likely be lower.

---

## Author

**Smita Sahu**
