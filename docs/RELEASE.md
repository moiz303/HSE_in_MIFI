# Запуск и релиз (Python 3.12)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -t . -v
```

После авторизации Kaggle: `python -m src.pipeline kaggle` скачает тест.
`python -m src.pipeline all` пересоздаст разбиение, обучит `artifacts/model.joblib`,
оценит её, соберёт этой же моделью CSV и сохранит чексуммы релиза. Holdout исключён.
Для готовой модели задайте одинаковый `--model` или `submission.model_file`:

```bash
python -m src.pipeline evaluate --model artifacts/model_sweep1.joblib
python -m src.pipeline submission --model artifacts/model_sweep1.joblib
python -m src.pipeline release --model artifacts/model_sweep1.joblib
```

Без выбора используется `artifacts/model.joblib`. Метрики включают MSE по SI,
метрики log10 и 95% bootstrap-интервалы MSE/RMSE по молекулам, не каркасным группам.
Блок `kaggle_train` отделяет строки соревнования от ChEMBL. Калибровка отклонена:
`reports/experiment_calibration.md`.

`reports/applicability_{validation,submission}.csv`: ID, максимальное сходство
Танимото с обучающими молекулами (Morgan radius=2, 2048 бит), флаг сходства < 0.3.
Флаг — диагностика, не вероятность ошибки; прогнозы и Kaggle CSV не меняются.
Новые модели хранят обучающие SMILES; старым sweep-моделям нужен именно исходный
`data/processed/train.csv`, который невозможно восстановить из их joblib.

`reports/release.json` содержит конфиг и SHA256 модели, данных, разбиения,
зависимостей, метрик, CSV и отчётов. Несогласованные результаты блокируют релиз.
Сохраните перечисленные файлы вместе с JSON: он не является архивом.
`reports/runs/*.json` хранит параметры и статус запусков.

После фиксации модели выполните `python -m src.pipeline holdout --model <путь>`,
затем `release` с той же моделью. Для baseline из `all`: `artifacts/model.joblib`.
Holdout проверяет пересечение молекул/каркасов, не переобучает модель и записывает
`reports/holdout.json`. Файл блокирует повтор, даже после ошибки; решение о повторе
принимает команда. `release` проверит его соответствие модели и добавит чексумму.
