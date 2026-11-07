# VACV: SMILES → индекс селективности

Актуальный путь соревнования: предсказывать **SI VACV только по SMILES**.
В описании Kaggle запрещены `CC50_uM` и `IC50_uM_VACV` для построения моделей
и предсказания; команда подтвердила строгую трактовку, включая обучающие цели.
Новый workflow использует из Kaggle только `SMILES` и `SI VACV`.
[Проверка требований](docs/current_protocol.md), [общий скореборд](reports/current/all_models.md),
[парное сравнение источников](reports/current/source_comparison.md).

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

Первоначальный OOF-кандидат и лучший официальный результат:
`chemeleon_domain_log1p__public_plus_kaggle`,
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

Последующие проверки не улучшили официальный результат:

| Кандидат | OOF MSE | Kaggle Score |
|---|---:|---:|
| CheMeleon SI domain log1p, public + Kaggle | 5061848.48 | **1586333.78789** |
| Он же, ограничение train-квантилем .99 | 5061848.48 | 1586412.96821 |
| Tanimoto GP, public + Kaggle, observation mean | **4973199.16** | 18558089.19131 |
| Hurdle GP power1, Kaggle-only, latent mean | 5070675.74 | 1949484.90648 |

Около 96% ошибки некоторых OOF-моделей создаёт одна строка с SI=36066.667.
Меньшая OOF MSE не доказывает улучшения скрытого теста.
[Независимый аудит](reports/current/protocol_audit.md) проверяет разбиения,
взвешенное GP-правдоподобие, влияние хвоста и интервалы неопределённости.
Цель ещё не достигнута. Четыре попытки исходного окна использованы;
ожидаемый сброс — 8 ноября в 03:00 по Москве, перед отправками квота проверяется через API.

Дополнительные блоки: публичные auxiliary-задачи на 4430 измерениях/1711 молекулах,
GP на CheMeleon/RBF и hurdle GP. Все используют прежние development-строки,
holdout остаётся закрыт. Лучший OOF mixed GP не продвигается после провала Kaggle.
Страты zero/bulk диагностируют риск, не подменяя raw MSE.
Завершены ещё 6 ансамблей (3 seed, преобразования SI log1p/Box-Cox .25/.5,
два источника). Лучший OOF5063287.05 не обошёл исходную голову5061848.48;
все шесть воспроизведены отдельно и не продвигаются на Kaggle.
Общая таблица содержит107 конфигураций и контролей; это89 конфигураций обучения
и18 дополнительных постобработок, а не107 независимых подтверждений.

Воспроизводимый read-only отчёт: `output/jupyter-notebook/vacv-model-comparison.ipynb`.
Обновление таблиц и выполнение всех кодовых ячеек отчёта:

```powershell
../env_arch/Scripts/python.exe -m src.current.scoreboard
../env_arch/Scripts/python.exe -m src.current.notebook_report
```

Текущий отдельный offline-эксперимент: `python -m src.current.bio_transfer`.
Сначала он обучает и замораживает предиктор только публичных IC50/EC50/CC50,
затем использует его выходы для SI-калибровки и сравнивает с теми же моделями
без биологических признаков. Оба режима источников, прежние18 folds,8 контролируемых
конфигураций. Это новый эксперимент, не подтверждённое улучшение; результаты
появятся в `reports/current_bio_transfer/`. Частичные fold-файлы защищены
fingerprint и не коммитятся; Kaggle не отправляется автоматически.

```powershell
../env_arch/Scripts/python.exe -m src.current.tail_controls --predict-artifact artifacts/current/chemeleon_domain_log1p__public_plus_kaggle.joblib --quantile 0.99 --out reports/current_tail/prediction.csv
```

TabPFN — отдельный эксперимент: `requirements-tabular.txt`,
`python -m src.current.tabular`. Авторские веса и атрибуция:
[Prior Labs TabPFN-v2](https://huggingface.co/Prior-Labs/TabPFN-v2-reg).
Контекст модели хранится отдельно от скачиваемых авторских весов.
Свежая проверка TabPFN пока не прошла: этот эксперимент исключён из кандидатов
на отправку до исправления воспроизводимости; допуск проверки не ослабляется.
Диагностика: `reports/current_tabular/reproducibility_audit.json`.
Изменение embedding на1.8e-7 и PCA-признаков на1.6e-5 заметно меняет некоторые
прогнозы TabPFN. Исторический исходный batch воспроизводится, отдельный batch
не проходит прежний допуск. Модели и прогнозы не переписаны для сокрытия расхождения.

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

План финального refit на всех320 конкурсных строках + прежних80 публичных:
`configs/current_release_refit.json`, `reports/current_release_refit/plan.json`.
Без флага модуль `python -m src.current.release_refit` только проверяет структуры
и печатает план, не читает резервные цели для обучения. Включение53 резервных
строк требует отдельного согласия пользователя; по умолчанию оно не дано.
Параметры уже выбранной модели фиксированы, новой оценки по резерву не будет.

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
