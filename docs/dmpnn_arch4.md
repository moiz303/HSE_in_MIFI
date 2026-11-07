# Architecture 4: shared D-MPNN with a direct SI head

Run the nested OOF experiment and final development refit with:

```bash
PYTHONPATH=/Users/f37/HSE_arch_env/lib/python3.12/site-packages:$PWD python3 -m src.dmpnn_workflow
```

The checked-in settings are in `configs/dmpnn_arch4.json`. The loader validates the committed split manifest; the fit receives only train and validation frames. It scores the original Kaggle training targets from `kaggle_reference.csv`; duplicate structures are count-weighted in the direct-SI loss. IC50 and CC50 use independent masks, exact log-MSE, and one-sided losses for censored bounds. Unknown or conflicting qualifiers are skipped. Inner scaffold folds choose auxiliary-loss weight and stopping epoch; outer OOF folds are not used for selection. The final artifact is refit on the 1007 development structures. Holdout endpoints and test endpoints are not used.

## Results

The seed 20261106 scaffold OOF scored 267 original Kaggle rows:

| Targets | Weighted XGBoost baseline | D-MPNN | Change |
|---|---:|---:|---:|
| All rows | 29,657,226.52 | 23,912,715.09 | -19.37% |
| Service-label rows | 99,643,915.00 | 49,691,463.00 | -50.13% |
| Measured rows | 6,676,522.73 | 15,448,051.60 | +131.38% |

The overall reduction is driven by lower error on the service-label subgroup; error on measured SI more than doubles. This is an OOF development comparison, not a Kaggle score, and it does not establish a model promotion. The existing default and baseline remain unchanged.

Artifacts: `reports/dmpnn_arch4_training.json`, `reports/dmpnn_arch4_oof.csv`, and `artifacts/dmpnn_arch4.pt`. The checkpoint records its training settings and runtime package versions.

## Дополнение из опубликованных измерений

Опциональный путь можно запустить флагом `--new-dataset`. Он оставляет те же outer folds и те же 267 оригинальных Kaggle targets для OOF. На обучающих частях используются только сырые CC50 наблюдения; EC50 остаётся отдельным endpoint и не передаётся IC50-голове. Цензурированные CC50 вносят односторонний bounded loss. Никакие дополненные SI-пары не создаются в train. Сначала удалены scaffolds исходного holdout, затем перед каждым inner/outer fit удаляются scaffolds его validation fold. Финальный refit добавляет все разрешённые CC50 наблюдения после holdout-фильтра.

Из 138 сохранённых исходных наблюдений (92 EC50 и 46 CC50) после глобального holdout-фильтра остаётся 126 (84 EC50 и 42 CC50). В final fit получается 1019 уникальных структур и 1049 записей обучения: 1007 исходных development структур плюс 42 CC50 измерения. По outer fold в обучение попадало 36, 38 и 30 supplementary CC50 наблюдений; структуры каждого отложенного fold исключались. Таблица пар на 184 контекста — воспроизводимый описательный файл, она не используется при обучении.

| Цели | Weighted XGBoost | D-MPNN | D-MPNN + 42 CC50 |
|---|---:|---:|---:|
| All 267 rows | 29,657,226.52 | 23,912,715.09 | **22,797,541.95** |
| 66 service-label rows | 99,643,915.00 | 49,691,463.00 | 49,838,463.13 |
| 201 measured rows | 6,676,522.73 | 15,448,051.60 | **13,918,433.51** |

Добавление CC50 снизило общую OOF MSE D-MPNN на 4.66%, а measured-subset MSE — на 9.90%; на service-label subset MSE выросла на 0.30%. Относительно weighted XGBoost полная ошибка ниже, но она остаётся заметно выше на measured subset. Это development OOF, а не Kaggle score и не решение продвигать эту модель в default.

Эксперимент сохраняет checkpoint и отчёт с суффиксом `_new_dataset`: `artifacts/dmpnn_arch4_new_dataset.pt`, `reports/dmpnn_arch4_oof_new_dataset.csv`, `reports/dmpnn_arch4_training_new_dataset.json`.
