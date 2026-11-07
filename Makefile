# Обёртка над python -m src.pipeline. На Windows make может быть не установлен —
# тогда запускаем те же команды напрямую, они перечислены в README.
PYTHON ?= python
CONFIG ?= configs/baseline.json

.PHONY: help data kaggle splits train eval submission release holdout test clean

help:
	@echo "data        выгрузка из открытых баз, валидация, стандартизация (A)"
	@echo "kaggle      скачать файлы соревнования (нужна авторизация)"
	@echo "splits      scaffold-разбиение и манифест                        (B)"
	@echo "train       обучение по конфигу $(CONFIG)                        (B C)"
	@echo "eval        метрики на validation с интервалами                  (B D)"
	@echo "submission  predictions -> reports/submission.csv                (D)"
	@echo "release     чексуммы выбранной модели и результатов            (D)"
	@echo "holdout     однократно после фиксации модели                     (D)"
	@echo "test        тесты и smoke-прогон                                 (все)"

data:
	$(PYTHON) -m src.pipeline data --config $(CONFIG)

kaggle:
	$(PYTHON) -m src.pipeline kaggle --config $(CONFIG)

splits:
	$(PYTHON) -m src.pipeline splits --config $(CONFIG)

train:
	$(PYTHON) -m src.pipeline train --config $(CONFIG)

eval:
	$(PYTHON) -m src.pipeline evaluate --config $(CONFIG)

submission:
	$(PYTHON) -m src.pipeline submission --config $(CONFIG)

release:
	$(PYTHON) -m src.pipeline release --config $(CONFIG)

holdout:
	$(PYTHON) -m src.pipeline holdout --config $(CONFIG)

test:
	$(PYTHON) -m unittest discover -s tests -t . -v

clean:
	rm -rf artifacts/*.joblib reports/metrics.json reports/submission.csv
