# Адаптация графового readout для прямого SI

Предыдущий концентрационный LoRA-опыт не обошёл равное продолжение голов
(`docs/chemeleon_finetune.md`). Его модели и конкурсные концентрационные цели
непереносимы в строгий протокол. Новый фактор — адаптация слоя `W_o` по
`log1p(SI VACV)`, включая все нули актуального `train_v2`.

Сравниваются четыре обучающих конфигурации: продолжение головы и LoRA rank8,
каждая с Kaggle-only и public+Kaggle. Нормировки, исходная доменная SI-голова,
порядок minibatch, случайные draws, head learning rate и40 дополнительных
эпох совпадают. Исходные frozen-контроли воспроизводятся на каждом фолде.
Разбиения прежние:267 development-строк,18 molecule/scaffold folds;53 резервные
строки закрыты. Две доменные SI-головы относятся к источникам, не к IC50/CC50.

`W_i/W_h` заморожены; это частичная адаптация атомной проекции, а не обучение
всей передачи сообщений. Atom-prefix кэш float32/memmap строится без меток.
Низкоранговая поправка применяется до ReLU и mean pooling; anchor ограничивает
сдвиг фактически нормированных/clipped признаков. Старый concentration CLI
не вызывается. Кэш производный и не коммитится.

```powershell
../env_arch/Scripts/python.exe -m src.current.graph_readout --pilot
../env_arch/Scripts/python.exe -m src.current.graph_readout
../env_arch/Scripts/python.exe -m src.current.graph_readout --verify
```

Пилот проверяет время и контроль на одном фолде, не выбирает модель.
Полный запуск сохраняет OOF, сравнение, четыре артефакта и submissions в
`reports/current_graph_readout/` и `artifacts/current_graph_readout/`.
Повторное выполнение использует только фолды с совпавшим отпечатком кода,
данных, параметров и кэша.

Заранее заданный фильтр: общая raw-SI MSE ниже frozen CheMeleon, zero/bulk MSE
не хуже1.2×, выиграны хотя бы2/3 scaffold-повтора. Для утверждения о пользе
LoRA она должна также обойти matched continuation. Эти исследовательские
условия не гарантируют Kaggle Score; автоматических отправок нет.
