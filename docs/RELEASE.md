# Запуск и релиз

Нужен Python 3.12. Обучающие данные и манифест разбиения лежат в репозитории.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -t . -v
```

После авторизации в Kaggle скачайте тест: `python -m src.pipeline kaggle`.
Для нового baseline запустите `python -m src.pipeline all`: он пересоздаст
разбиение, обучит `artifacts/model.joblib`, оценит её, соберёт CSV этой же
моделью и запишет чексуммы релиза. Holdout в `all` не входит.

Для готовой модели используйте один конфиг с `submission.model_file` либо
одинаковый аргумент `--model` во всех командах:

```bash
python -m src.pipeline evaluate --model artifacts/model_sweep1.joblib
python -m src.pipeline submission --model artifacts/model_sweep1.joblib
python -m src.pipeline release --model artifacts/model_sweep1.joblib
```

Без явного выбора используется `artifacts/model.joblib`. Метрики содержат
MSE по SI, метрики в log10 и 95% bootstrap-интервалы MSE и RMSE log10.
Вложенный блок `kaggle_train` считает те же метрики только по строкам источника
соревнования, отдельно от хвоста ChEMBL.
Интервалы получены повторной выборкой молекул, а не каркасных групп.
Калибровка уже проверена и отклонена: `reports/experiment_calibration.md`.

`reports/applicability_validation.csv` и `applicability_submission.csv`
содержат ID, максимальное сходство Танимото с обучающими молекулами и
`outside_domain`. Фингерпринт — бинарный Morgan radius=2, 2048 бит.
Сходство < 0.3 — диагностическое предупреждение, не оценка вероятности ошибки;
оно не меняет прогноз и формат Kaggle CSV. У новых pipeline-моделей обучающие
SMILES хранятся внутри артефакта. Для старых моделей sweep нужен именно тот
`data/processed/train.csv`, на котором они обучались: восстановить его из
старого joblib автоматически нельзя.

`reports/release.json` фиксирует конфигурацию и SHA256 модели, данных,
разбиения, зависимостей, метрик, CSV и отчётов. Несовпадение модели/конфига
между оценкой и сабмитом останавливает сборку релиза. Сохраните перечисленные
файлы вместе с этим JSON; сам JSON не является архивом артефактов.
`reports/runs/*.json` хранит параметры и статус каждого запуска.

Только после фиксации финальной модели выполните отдельную команду:

```bash
python -m src.pipeline holdout --model artifacts/model_sweep1.joblib
python -m src.pipeline release --model artifacts/model_sweep1.joblib
```

Для baseline из `all` укажите `--model artifacts/model.joblib`. Holdout не
переобучает модель, проверяет пересечение обучающих и holdout-молекул/каркасов
и пишет `reports/holdout.json`. Наличие этого файла блокирует повтор даже
после ошибки. Не удаляйте его для подбора модели; решение о повторе принимает
команда. После проверки `release` включит holdout в чексуммы.
