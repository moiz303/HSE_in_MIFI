# Запуск и сборка сабмита

Нужен Python 3.12. Обучающие данные лежат в репозитории.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -t . -v
```

Скачайте тест соревнования после авторизации в Kaggle:

```bash
python -m pip install kaggle
kaggle auth login
python -m src.pipeline kaggle
python -m src.pipeline all
```

`all` создаёт разбиение, обучает модель, считает метрики и сохраняет
`reports/submission.csv`. Для сборки из готовой модели:

```bash
python -m src.pipeline submission
```

Перед сохранением проверяются колонки, число строк и конечность прогнозов.
В `reports/runs/*.json` сохраняются команда, конфигурация (включая seed),
чексуммы входов, время и статус запуска. Версии зависимостей — в
`requirements.txt`. `run_mac.sh` также сохраняет консольный лог обучения.

Для отчёта сохраните модель, манифест разбиения, метрики и JSON запуска.
Holdout проверяется один раз после заморозки модели по решению команды.
