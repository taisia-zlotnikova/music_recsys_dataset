# Music by Prompt v2

Синтетические музыкальные диалоги по сессиям Music4All-Onion. Скрипты просмотра и статистики: Python 3.9+, без сторонних библиотек.

Обучение encoder–decoder: [команды и формат данных](twotower/README.md).

## Данные

- `data/dialogs_v2/` — исходные `part-XXX.jsonl.xz`.
- `data/music4all/` — `id_information.csv`, `id_metadata.csv`, `id_genres.csv`, `id_tags.csv`, `id_lang.csv` из Music4All.

Одна JSONL-строка — целый диалог. `tracks` содержит ID Music4All; каждый ход `turns` имеет вид `[user_lines, k]` и получает следующие k треков из списка. `ts_start` и `ts_end` — начало и конец сессии. Порядок строк не хронологический.

## Команды

Запускать из корня проекта:

```bash
python3 scripts/stats.py                       # статистика clean_v2=true
python3 scripts/cards.py                       # карточки по точному track_id
python3 scripts/show_sample.py --min-turns 2    # пример с карточками
python3 scripts/split.py                       # воспроизвести train/val
```

Статистика: [reports/report.html](reports/report.html). Карточки записываются в `reports/cards.jsonl`; доступны для всех 32 462 треков. Просмотр использует случайную выборку из `reports/sample.jsonl`.

## Итоговое разбиение

Включены только `clean_v2=true`. Граница **2018-08-06 00:00 UTC**:

| Часть | Правило | Диалоги | Доля |
|---|---|---:|---:|
| Train | `ts_end < граница` | 44 029 368 | 90,01% |
| Val | `ts_start >= граница` | 4 884 642 | 9,99% |

**139 пересекающих границу диалогов исключены целиком.** Один диалог не делится между частями; пользователь может присутствовать в обеих.

Готовые файлы: `data/split_v2/train.jsonl.gz` и `data/split_v2/val.jsonl.gz`. Это исходные записи JSONL, сжатые gzip, без добавления метаданных Music4All. Доли рассчитаны среди сохранённых диалогов.

`data/`, `_archive/` и `reports/cards.jsonl` исключены из Git; исходные данные и готовые части передаются отдельно.
