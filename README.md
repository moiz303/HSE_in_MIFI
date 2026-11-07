# VACV: SMILES → индекс селективности

Актуальный путь соревнования: предсказывать **SI VACV только по SMILES**.
В описании Kaggle запрещены `CC50_uM` и `IC50_uM_VACV` для построения моделей
и предсказания; команда подтвердила строгую трактовку, включая обучающие цели.
Новый workflow использует из Kaggle только `SMILES` и `SI VACV`.
[Проверка требований](docs/current_protocol.md), [таблица результатов](reports/current/comparison.md).

## Данные и сравнение

- Источник целей — официальный `train_v2.csv`: 320 строк, 83 нуля. Старые 81 цели
  10000 заменены организаторами на 0; контрольная сумма закреплена в конфигурации.
- Development: 267 исходных строк, 66 нулей. Остальные 53 не обучались и не оценивались;
  из них две структуры были исключены ещё в старой сборке, а не являются новым holdout.
- `data/current/public_dataset.csv`: 109 точных пар IC50/CC50, 87 молекул;
  те же статья и известная клеточная линия, явные единицы и DOI.
- Для смешанного обучения доступны 80 публичных пар, 63 дополнительные молекулы.
  Пересечения с конкурсными структурами и резервными каркасами исключены;
  цензура, EC50/GI50/TC50 и неподтверждённые пары не входят в этот набор.
- Файлы фактических когорт: `data/current/kaggle_development.csv` и
  `data/current/development_mixed.csv`. Конкурсные цели не усредняются с публичными.

Обучены **14 вариантов**: Ridge, общий и доменный MLP на замороженном авторском
CheMeleon, прямой CatBoost; для MLP/domain/CatBoost также проверен `log1p(SI)`.
Каждый — Kaggle-only и public+Kaggle, на одинаковых 18 molecule/scaffold OOF-фолдах.
Параметры и веса источников: `configs/current.json`.

Выбранный по OOF кандидат: `chemeleon_domain_log1p__public_plus_kaggle`,
средний MSE **5061848.48**. Лучший дополнительный endpoint-диагностический MSE:
CatBoost log1p на смеси, **1968835.32** на 468/597. Это другой неполный эталон,
он не использован для выбора. После отдельного разрешения новый SI-кандидат
получил официальный **Kaggle Score 1586333.78789**, ref 58103006.
Текущая цель — top10 или Score<800000. История попыток:
`reports/current/kaggle_goal_submissions.json`.
Улучшение выбранного метода к нулю в OOF около 1.77%; exploratory bootstrap-интервал
включает отсутствие выигрыша. Это не независимое подтверждение превосходства.

Дополнительно проверены 25 вариантов Tanimoto-kernel/аналогов/ExtraTrees/бустинга
и 3 варианта Prior Labs TabPFN-v2 с PCA CheMeleon + RDKit. Их OOF не обошёл
основной SI-кандидат: `reports/current_search/` и `reports/current_tabular/`.
Контроли квантильных ограничений и равновесных ансамблей: `reports/current_tail/`.
Ограничение по train-квантилю .99 сохраняет все OOF-прогнозы, ограничивая три
экстремальных финальных прогноза; официальный результат проверяется отдельно.

```powershell
../env_arch/Scripts/python.exe -m src.current.tail_controls --predict-artifact artifacts/current/chemeleon_domain_log1p__public_plus_kaggle.joblib --quantile 0.99 --out reports/current_tail/prediction.csv
```

TabPFN — отдельный эксперимент: `requirements-tabular.txt`,
`python -m src.current.tabular`. Авторские веса и атрибуция:
[Prior Labs TabPFN-v2](https://huggingface.co/Prior-Labs/TabPFN-v2-reg).
Контекст модели хранится отдельно от скачиваемых авторских весов.

## Запуск

Используйте Python3.12 и отдельное окружение. На Windows проверено:

```powershell
../env_arch/Scripts/python.exe -m pip install -r requirements-current.txt
kaggle auth login
kaggle competitions download molecular-activity-prediction-hack -f train_v2.csv -p data/raw/kaggle
kaggle competitions download molecular-activity-prediction-hack -f test.csv -p data/raw/kaggle
kaggle competitions download molecular-activity-prediction-hack -f sample_submit.csv -p data/raw/kaggle
../env_arch/Scripts/python.exe train_model.py --stage all
../env_arch/Scripts/python.exe -m src.current.results
../env_arch/Scripts/python.exe -m src.current.verify
```

На Mac: `./run_mac.sh` (Python3.12, отдельная `.venv-current`, тот же протокол).
Mac-скрипт проверен по синтаксису; фактическое обучение выполнено на Windows CPU.
Авторские веса CheMeleon скачиваются по фиксированному URL с проверкой MD5/SHA256;
сами веса и структурные кэши не коммитятся.

Обученные модели: `artifacts/current/`. Выбранный кандидат и команда инференса:
`reports/current/best_candidate.json`; все файлы submission: `reports/current/`.

```powershell
../env_arch/Scripts/python.exe train_model.py --predict-artifact artifacts/current/chemeleon_domain_log1p__public_plus_kaggle.joblib --out reports/current/prediction.csv
```

Для параллельной работы запустите `--stage train --worker-repeat 0`, `1`, `2`
в отдельных терминалах, затем `--stage train --final-only`. Возобновление проверяет
конфигурацию, код, данные и веса энкодера. Test endpoint-диагностика запускается
отдельным `src.test_diagnostics` только после сохранения моделей и прогнозов;
она не пересчитывает официальный Kaggle Score и не меняет OOF-выбор.

## Сбор публичных данных

```powershell
../env_arch/Scripts/python.exe -m src.current.collect
```

Создаётся новый неизменяемый снимок в `data/public_downloads/<UTC>/`:
нормированные измерения, точные пары с `smiles,IC50,CC50,SI,doi`, исходные ID,
журналы исключений и контрольные суммы. Запросы идут только к целям VACV типа
ORGANISM в [ChEMBL API](https://www.ebi.ac.uk/chembl/api/data/), с HTTP-кэшем.
Это отдельный сбор; он не перезаписывает используемый для сравнения снимок.
Перенос нового снимка в эксперимент требует новой версии сборки и проверки.

## Проверки и история

```powershell
../env_arch/Scripts/python.exe -m unittest discover -s tests -t . -v
```

Проверяются нули, единицы, типы endpoint, разбиения, запрет конкурсных концентраций,
неотрицательные прогнозы и переносимость артефактов. Все14 моделей воспроизведены
в новом процессе на597 SMILES; исходный CatBoost из testing_main повторяется точно.

`legacy/` содержит старые entry points и документацию; 31 старый корневой лог
перенесён в `reports/archive/logs/`. Старые модели и модули их загрузки сохранены.
`data/processed/` — исторический снимок: новый workflow читает из него только
структуры для неизменных ролей разбиения, а не старые SI или концентрации.
Новый датасет находится в `data/current/` и защищён контрольными суммами.

Исторический public Score 1614079.37878 относится к концентрационным доменным
головам, использовавшим запрещённые по текущей трактовке конкурсные endpoints.
Такая модель исключена из нового рабочего протокола; её оценка не переносится
на новые SI-головы. [Исторический скореборд](reports/archive/model_scoreboard_v1.md).
