# Чистый запуск и релиз

## Чистый запуск

Нужен Python 3.12. Тренировочные данные и конфигурация уже лежат в репозитории;
тестовый файл соревнования в Git не хранится.

```bash
git clone https://github.com/moiz303/HSE_in_MIFI.git
cd HSE_in_MIFI
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -t . -v
```

CI запускает те же тесты на небольших синтетических данных. Тестам не нужны
учётная запись Kaggle, файл теста соревнования или полное обучение.

## Сборка сабмита

Установите Kaggle CLI, войдите в аккаунт и скачайте файлы соревнования один раз:

```bash
python -m pip install kaggle
kaggle auth login
python -m src.pipeline kaggle
```

Затем запустите весь baseline-процесс:

```bash
python -m src.pipeline all
```

Команда пересоздаёт scaffold-разбиение, обучает модель по конфигурации,
считает метрики на train и validation и пишет `reports/submission.csv`.
Для повторной сборки из уже обученной модели используется
`python -m src.pipeline submission`. Она требует сохранённую модель и
`data/raw/kaggle/test.csv`.

## Что сохраняется

Каждый запуск `src.pipeline` и `train_model.py` получает пару файлов в
`reports/runs/`:

- `run_<время>.json` — команда, конфигурация, seed, версии библиотек, хеши
  входных файлов и статус запуска;
- `run_<время>.log` — вывод команды, включая ошибки.

Не добавляйте Kaggle credentials в репозиторий. Модель и CSV сабмита
игнорируются Git; логи и JSON запуска можно приложить к отчёту, чтобы
зафиксировать происхождение результата.

## Перед отправкой

1. Убедитесь, что CI зелёный.
2. Проверьте, что `reports/submission.csv` создан и содержит колонки `index`
   и `SI VACV`, по строке на каждую молекулу из теста.
3. Сохраните JSON и лог соответствующего запуска вместе с отчётом.
4. Загружайте файл в соревнование:

```bash
kaggle competitions submit -c molecular-activity-prediction-hack \
  -f reports/submission.csv -m "SMILES baseline"
```

Обычный `all` оценивает модель только на train и validation. Holdout не
используйте для выбора модели, порога или параметров; его однократная проверка
проводится отдельно в конце по решению команды.
