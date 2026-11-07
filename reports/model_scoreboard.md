# Скореборд

Актуальное сравнение train_v2 в строгом SMILES → SI протоколе:
[reports/current/comparison.md](current/comparison.md),
[CSV](current/comparison.csv), [эффект публичных данных](current/public_effect.csv).

Проверены 42 конфигурации обучения на одинаковых 18 OOF-разбиениях и 18
дополнительных контролей хвоста/ансамблей. Лучший новый официальный
Score: **1586333.78789**, SI-доменные головы CheMeleon, ref 58103006.
Ограничение train-квантилем .99: **1586412.96821**, ref 58103262; улучшения нет.

Текущая цель — top10 или Score<800000; из четырёх разрешённых попыток использованы две.
[Реестр заявок](current/kaggle_goal_submissions.json),
[ядра/бустинг/деревья](current_search/comparison.csv),
[TabPFN](current_tabular/comparison.csv),
[ограничения хвоста](current_tail/comparison.csv).

[Исторические23 модели и их прежние scores](archive/model_scoreboard_v1.md)
сохранены отдельно. Концентрационные головы с конкурсными CC50/IC50 не входят
в нынешний строгий протокол и не подменяют новые SI-модели.
