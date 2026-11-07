# Скореборд

Актуальное сравнение train_v2 в строгом SMILES → SI протоколе:
[reports/current/comparison.md](current/comparison.md),
[CSV](current/comparison.csv), [эффект публичных данных](current/public_effect.csv).

Полная актуальная таблица всех конфигураций и проверок:
[all_models.md](current/all_models.md), [all_models.csv](current/all_models.csv).
Все используют одинаковые 18 OOF-разбиений. Лучший новый официальный
Score: **1586333.78789**, SI-доменные головы CheMeleon, ref 58103006.
Ограничение train-квантилем .99: **1586412.96821**, ref 58103262; улучшения нет.

Mixed GP с лучшей OOF4973199.16 получил Kaggle18558089.19131 и не продвигается.
Kaggle-only hurdle получил1949484.90648. Цель — top10 или Score<800000;
все четыре исходные попытки использованы. Сброс ожидается8ноября03:00МСК.
[Реестр заявок](current/kaggle_goal_submissions.json),
[ядра/бустинг/деревья](current_search/comparison.csv),
[TabPFN](current_tabular/comparison.csv),
[ограничения хвоста](current_tail/comparison.csv).

[Исторические23 модели и их прежние scores](archive/model_scoreboard_v1.md)
сохранены отдельно. Концентрационные головы с конкурсными CC50/IC50 не входят
в нынешний строгий протокол и не подменяют новые SI-модели.
