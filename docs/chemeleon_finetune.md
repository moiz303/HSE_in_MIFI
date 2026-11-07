# CheMeleon: частичное дообучение и аудит данных

Конфигурация зафиксирована до тестовой диагностики: `configs/chemeleon_finetune.json`.
Окружение Python 3.12: `../env_arch`, зависимости `requirements-architectures.txt`.

```powershell
../env_arch/Scripts/python.exe -m src.oof_error_audit
../env_arch/Scripts/python.exe -u -m src.chemeleon_finetune
../env_arch/Scripts/python.exe -m unittest discover -s tests -t . -v
```

Для параллельного CPU-запуска можно выполнить `--worker-repeat 0`,
`--worker-repeat 1`, `--worker-repeat 2` в трёх терминалах. Затем обычная
команда собирает завершённые fold-файлы и выполняет финальное обучение.
Возобновление проверяет отпечаток конфигурации, данных и кода; workers не
выбирают модель и не пишут финальные артефакты.

## Что сравниваем

Шесть вариантов: `frozen`, `continue`, `lora`, `original_warm`,
`original_continue`, `original_lora`. Контроль `continue` получает столько же
дополнительных эпох голов, сколько LoRA. Базовая голова: 160 эпох; продолжение:
40 эпох, batch32. LoRA rank8/alpha8 обновляет финальную атомную проекцию `W_o`,
совместно с двумя головами IC50/CC50. Это частичное дообучение энкодера;
передача сообщений `W_i/W_h` остаётся замороженной. Их выходы кэшируются без
меток, чтобы обучение на CPU было практически выполнимым.

Нулевая LoRA воспроизводит исходные признаки. Обновление вычисляется как
разность ReLU до/после изменения проекции с последующим mean pooling.
Регуляризатор удерживает фактические нормированные, ограниченные признаки
головы около исходных. Нормировки обучаются только внутри fit-части.

## Аудит исходных измерений

Аудит относится к TRAIN+validation, не к тестовым концентрациям. 201 исходная
строка имеет измеренный SI; 6 OOF-прогнозов на строку, без объявления их
независимыми наблюдениями. Одна молекула с SI=36066.667 создаёт 92.41% суммарной
измеренной OOF-ошибки; пять — 98.45%. Её исходные IC50=0.03 и CC50=1082 мкМ
совпадают с объединённой записью, поэтому удаление или ручная замена не обоснованы.
В исходном экспорте крупнейшей ошибки нет DOI. Найдена вероятная первичная
публикация: [Sokolova et al., ChemMedChem 2022](https://doi.org/10.1002/cmdc.202100771),
[авторская рукопись, таблица 1](https://www.researchgate.net/publication/359789665_Design_Synthesis_and_Biological_Evaluation_of_-Camphor-_and_--Fenchone-Based_Derivatives_as_Potent_Orthopoxvirus_Inhibitors).
Для соединения 8e указаны CC50=1082.0±233.6 мкМ, IC50=0.03±0.01 мкМ,
но SI=31500, тогда как отношение округлённых концентраций равно 36066.667.
Это кандидат на расхождение опубликованного SI и отношения округлённых чисел;
идентичность конкретной стереоструктуры требует проверки рисунка/приложения.
Ни метки, ни данные обучения по этой находке не изменяются. Подмена SI
публикационным числом исказила бы сравнение с исходными Kaggle TRAIN целями.

Публичные знаки цензуры маскировали доступные точные оригинальные концентрации.
Проверяем единое правило: по каждой голове точное оригинальное измерение
имеет приоритет, при его отсутствии остаётся публичная запись. Доступные
targets: 465/682 → 486/701. Пропуски не выводятся из SI; правило применяется
только к fit-части, без изменения валидационных целей.

## Выбор и результаты

Прежние molecule/scaffold OOF: 3 fold × 3 повторения. Выбор — среднее отношение
MSE по измеренным исходным SI к `frozen`, улучшение минимум 1%, победа в двух
из трёх повторов каждого протокола. Допускается максимум 2% ухудшения scaffold
MSE и 20% ошибки SI<=100. Иначе выбирается `frozen`. Holdout не используется.
Это исследовательский выбор на многократно использованных OOF.

`reports/finetune/selection.json` записывается до финального обучения и чтения
тестовых концентраций. После обучения все шесть файлов можно отдельно проверить:

```powershell
../env_arch/Scripts/python.exe -m src.test_diagnostics --submission reports/finetune/submission_frozen.csv reports/finetune/submission_continue.csv reports/finetune/submission_lora.csv reports/finetune/submission_original_warm.csv reports/finetune/submission_original_continue.csv reports/finetune/submission_original_lora.csv --output-prefix reports/finetune_test_diagnostics
../env_arch/Scripts/python.exe -m src.finetune_results
```

Таблица: `reports/finetune/comparison.md`; модели: `artifacts/finetune/`.
Повторный инференс читает только ID+SMILES:

```powershell
../env_arch/Scripts/python.exe -m src.chemeleon_finetune --predict-artifact artifacts/finetune/lora.joblib --out reports/finetune_prediction.csv
```

Официально лучшая модель и её подтверждённый Score 1750683.92577 сохраняются.
Ни одна новая заявка этим запуском не отправляется.
