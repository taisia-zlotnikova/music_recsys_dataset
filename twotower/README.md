# Двухбашенная модель

Результаты пилота и локальных проверок: [RESULTS.md](RESULTS.md).

## Текущий эксперимент v2: LoRA, один этап

Готовый Colab: [colab_lora.ipynb](colab_lora.ipynb). В нём четыре ячейки: загрузить архив кода,
установить зависимости, подключить Drive/скопировать данные, запустить три режима.
GitHub для этого не нужен. На Drive уже достаточно `music_prompt/data/features.npz`
и `music_prompt/data/pilot/` целиком; вариант `data/model/` тоже поддерживается.

**С первого шага вместе обучаются LoRA, таблица тегов и проекции. Постепенного размораживания нет.**
Исходные веса Qwen, двух слоёв энкодера истории и семи cross-attention остаются замороженными.
В q/k/v/o всех трёх частей добавлена LoRA rank=8, alpha=16, dropout=0.05. Gates обучаются;
их нулевая инициализация открывает путь в историю после первых обновлений, отдельной фазы не нужно.
Готовые MuQ-128 и Qwen-векторы названий-1024 заморожены, теги-256 обучаются с нуля.
Общая башня треков используется и в истории, и в каталоге; retrieval-вектор имеет размер 256.
`both`: **10 212 616** обучаемых параметров (из них 3 031 040 LoRA и 5 666 816 таблица тегов).

```bash
# Из корня репозитория, после установки зависимостей и копирования данных:
python -u -m twotower.experiment --out /path/on/drive/runs/lora_v2_e2 --hours 22 --batch-size 8
```

На T4 начинаем с batch=8, на A100 — 32; accumulation автоматически даёт эффективный batch=128.
Pilot: 50 000 train-диалогов / 82 122 хода, 5000 val-диалогов / 8421 ход.
История — 64 уникальных трека, текст — 128 токенов. E=2, cross_every=4.
AdamW: LoRA lr=1e-4, теги/проекции/gates lr=1e-3, weight decay=0.01, gradient clipping=1,
FP16 + GradScaler, gradient checkpointing. Loss — multi-positive full softmax по 32 462 трекам.

Порядок работы:
1. По 20 шагов `both`, `text`, `history`: замер скорости, сохранение checkpoint. Эти шаги не теряются.
2. По замеру выбираются общие **2 или 4 эпохи**. Расчёт оставляет 40% запаса к времени обучения
   и час на финальную оценку. Если две эпохи не помещаются, длинное обучение не запускается.
3. По одной эпохе каждого режима по очереди. Одна эпоха — 642 optimizer steps, четыре — 2568 на режим.
   Это продолжение с optimizer/RNG из checkpoint, а не повторная инициализация модели.
4. Validation на фиксированных 5000 ходах раз в эпоху и после начального замера.
   Выбирается best по NDCG@100. Все три best и обе эвристики оцениваются на всех 8421 val-ходах.

В выходной папке: `experiment.json`, `plan.json`, `both/`, `text/`, `history/`,
`baselines_all_val.json` и итоговый `results.json`. В каждой папке режима — `config.json`,
`metrics.jsonl`, `last.pt`, `best.pt`, после итоговой оценки `evaluation.json`.
Лог validation также содержит значения gates. Это пилотный эксперимент одного seed,
без гарантии сходимости и без гарантии превосходства над recency.

При разрыве Colab повторить команду с **той же выходной папкой и batch**.
`last.pt` сохраняется каждые 100 шагов: до 99 несохранённых шагов могут повториться.
Лимит `--hours` относится к текущему вызову; повторный запуск даёт новый бюджет.
План фиксируется в `plan.json`; автоматически с двух эпох на четыре при resume не переключается.
После смены A100 на T4 batch=32 может не поместиться: точный resume не меняет batch автоматически.

Для отдельного управляемого запуска:

```bash
python -u -m twotower.train --device cuda --tuning adapters --lora-r 8 \
  --mode both --encoder-layers 2 --cross-every 4 --batch-size 8 --accumulation 16 \
  --steps 2568 --eval-every 642 --save-every 100 --lr 0.001 --lora-lr 0.0001 \
  --out runs/lora_manual_both
```

Старые checkpoint не переводятся в новый режим: `--tuning legacy` (прежний default)
сохраняет старую архитектуру обучения. Там `--lora-r` добавлял адаптеры только в decoder.
Новый эксперимент всегда явно передаёт `--tuning adapters`. Старое обучение ниже описано для
воспроизводимости; для текущего Colab используй notebook выше.

## Подготовка данных и прежние команды

Запускать из корня репозитория, Python 3.10+.

```bash
pip install -r twotower/requirements.txt
python -m twotower.features --recsys ../recsys
python -m twotower.prepare
python -m twotower.show --index 0
```

`features` собирает все 32 462 ID из `reports/tracks.csv`: готовые MuQ-128, Qwen-1024 («исполнитель — название»), ID тегов и веса. Пересчёта нейросетевых признаков нет. Теги/жанры нормализуются; редкие (<5 треков) исключаются, остаются до 32 на трек. Веса Last.fm /100 и вес жанра 0.5, затем нормализация суммы. Общий файл `data/model/features.npz` можно отдать Паше. Внутри есть `track_ids`, `audio`, `text`, `tag_vocab`, `tag_ids`, `tag_weights`, `artists`, `titles`; строка в массивах всегда соответствует `track_ids`.

Если `reports/tracks.csv` ещё нет, его создаёт `python3 scripts/stats.py`. Для обучения достаточно готового `features.npz`: исходные parquet и соседний проект нужны только для его сборки.

## Выбор признаков трека

Простые baselines без GPU: `python -m twotower.baselines --out runs/baselines_5000.json` оценивает популярность и последние треки истории на тех же 5000 val-ходах (seed=42); `--val-turns 0 --out runs/baselines_all_val.json` — на всех подготовленных val-ходах. Популярность считается только по train-целям; recency использует историю готового примера и дополняет выдачу популярными. Метрики вычисляются той же функцией, что у моделей. В JSON сохраняются hash данных и выбранных индексов для проверки сопоставимости.

Текущий набор — исходный baseline, а не доказанно лучший. Старые результаты не изолируют вклад отдельных каналов. Выбор задаётся одним аргументом и действует одновременно на историю и кандидатов, поскольку у них общая TrackTower:

```bash
python -m twotower.train --item-features audio,text,tags --out runs/all_features
python -m twotower.train --item-features text,tags --out runs/no_audio
python -m twotower.train --item-features audio,text --out runs/no_tags
python -m twotower.train --item-features audio,tags --out runs/no_title
```

`text` здесь — готовое представление исполнителя/названия трека. Аргумент `--mode text` имеет другой смысл: отключает историю пользователя. Отключённые каналы вообще не создаются в TrackTower, итоговый retrieval-вектор по-прежнему имеет размерность `--dim` (256).

Чтобы добавить lyrics или заменить текстовый эмбеддер, нужны готовые векторы в NPZ с двумя массивами: `track_ids [N]` и `vectors [N,D]`. Затем:

```bash
# Дополнительный канал. ID из файла сопоставляются с основным каталогом, порядок может отличаться.
python -m twotower.features --base data/model/features.npz --add lyrics=data/lyrics_vectors.npz --out data/model/with_lyrics.npz
python -m twotower.train --features data/model/with_lyrics.npz --item-features audio,text,tags,lyrics --out runs/with_lyrics

# Замена Qwen-названия другим готовым представлением текста:
python -m twotower.features --base data/model/features.npz --add text=data/new_text_vectors.npz --out data/model/new_text.npz
```

Это присоединение готовых векторов, не их вычисление нейросетью. Размерность каждого канала определяется из массива; менять модель не нужно. Для текстового представления тегов можно добавить канал `tags_text`, включить его в `--item-features` и исключить обучаемый `tags`.

Пересобирать диалоги/историю при смене признаков не нужно: в подготовленной папке `track_ids.npy` фиксирует каталог и порядок индексов. В существующие локальные `pilot/` и `smoke/` этот файл уже добавлен после проверки исходного hash. Старые переносимые данные без него принимают только исходный features.npz. При новом наборе признаков начинаем отдельное обучение в новом `--out`; `--resume` требует прежних признаков и каналов. Выбор сохраняется в checkpoint и автоматически восстанавливается при оценке/просмотре.

`prepare` по умолчанию равномерно выбирает **50 000 train-диалогов и 5 000 val-диалогов**, seed=42. Читает существующие временные split целиком: их строки сгруппированы по трекам, брать начало для эксперимента нельзя. Затем делает один проход по исходному Onion (`_archive/userid_trackid_timestamp.tsv.bz2`). События выбранных пользователей сортируются в SQLite на диске; весь Onion не помещается в RAM. Это основная стоимость подготовки, даже для небольшого пилота. Нужны свободное место под временную базу и время на чтение архива. База удаляется после успешной подготовки.

Один пример — ход: пользовательские реплики до текущей включительно → все k целевых треков. В память истории попадают последние 64 уникальных трека из **всех доступных Onion-событий строго до начала сессии**, затем выдачи только предыдущих ходов. У событий на границе `timestamp == session_start` доступа нет. Для прошлых plays сохраняются log2-корзины количества и возраста в днях; для предыдущих выдач возраст=0, count увеличивается на 1 за показ. Валидация может использовать более ранние реальные прослушивания из val, но веса на них не обновляются. Текст содержит только пользовательские реплики: ответов ассистента в исходных данных нет. Последние токены контекста сохраняются, EOS добавляется явно.

В `data/model/pilot/` — mmap-шарды train/val, тексты примеров и manifest. `--train-dialogs 0 --val-dialogs 0 --out data/model/full` готовит весь корпус; это десятки миллионов ходов и значительно больше диска/времени. `--scan-limit` — только отладка на первых строках, такая выборка смещена и не подходит для отчётных метрик.

## Обучение в Colab T4

Готовый notebook: [открыть в Colab](https://colab.research.google.com/github/taisia-zlotnikova/music_recsys_dataset/blob/research/encoder-decoder-embeddings/twotower/colab.ipynb). В нём код клонируется из этой ветки, данные копируются с Drive, результаты сохраняются обратно на Drive. Первый запуск — 10 проверочных шагов.

Перенести репозиторий, `features.npz` и папку `pilot/`. Соседний recsys, исходные split и Onion в Colab уже не нужны. Установка зависимостей та же. Первое скачивание Qwen идёт из Hugging Face; для локального кеша есть `--local-files-only`.

Минимальная папка на Google Drive:

```text
MyDrive/music_prompt/
  twotower/                 # код из этой ветки, включая requirements.txt
  data/model/features.npz
  data/model/pilot/          # целиком, с manifest и train/val
```

В Colab включить GPU. Подключить Drive, скопировать код/данные на локальный диск среды (mmap быстрее, чем с Drive):

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
!mkdir -p /content/music_prompt/data/model
!cp -r /content/drive/MyDrive/music_prompt/twotower /content/music_prompt/
!cp /content/drive/MyDrive/music_prompt/data/model/features.npz /content/music_prompt/data/model/
!cp -r /content/drive/MyDrive/music_prompt/data/model/pilot /content/music_prompt/data/model/
%cd /content/music_prompt
%pip install -r twotower/requirements.txt
%pip uninstall -y torchvision torchaudio
```

Torchvision/torchaudio нам не нужны: аудиопризнаки уже посчитаны. Удаляем эти предустановленные пакеты в Colab, чтобы они не конфликтовали с закреплённым torch (например, `operator torchvision::nms does not exist`). После установки, если Colab попросит, перезапустить сессию Python и снова выполнить mount/cd. Проверочный запуск с сохранением на Drive:

```python
!python -m twotower.train --device cuda --steps 10 --eval-every 10 --val-turns 500 --out /content/drive/MyDrive/music_prompt/runs/sanity
```

Основной пилот на **две эпохи = 1284 optimizer steps**:

```python
!python -m twotower.train --device cuda --steps 1284 --out /content/drive/MyDrive/music_prompt/runs/encdec
```

На Drive в выбранном `--out` появятся `config.json`, `metrics.jsonl`, `best.pt`, `last.pt`. Checkpoint записывается при каждой validation и на последнем шаге. Продолжение после обрыва: та же команда с `--resume /content/drive/MyDrive/music_prompt/runs/encdec/last.pt`. Исходная Qwen загружается из Hugging Face заново, если кеш Colab утрачен. Новый запуск требует нового `--out`; для существующего используется `--resume`.

```bash
# Проверить память и скорость: 10 optimizer steps, эффективный batch 128.
python -m twotower.train --out runs/sanity --steps 10 --eval-every 10 --val-turns 500

# Первый эксперимент: encoder=2, cross-attention через 4 слоя Qwen.
python -m twotower.train --out runs/encdec --steps 1000

# Продолжить ТОТ ЖЕ запуск до 2000 шагов, сохранить прежние настройки.
python -m twotower.train --out runs/encdec --resume runs/encdec/last.pt --steps 2000

# Необязательный второй этап: добавить LoRA к Qwen, новый optimizer.
python -m twotower.train --out runs/encdec_lora --init runs/encdec/best.pt --lora-r 16 --steps 1000

python -m twotower.show --index 0 --checkpoint runs/encdec/best.pt

# Итоговые метрики лучшего checkpoint на всех подготовленных val-ходах.
python -m twotower.train --evaluate runs/encdec/best.pt --val-turns 0
```

Новые части: проекции/MLP трека, таблица тегов с нуля, проекции истории, два скопированных слоя Qwen и gated cross-attention. Исходные MuQ/Qwen-векторы и базовая Qwen заморожены; `--lora-r 16` обучает также адаптеры декодера. Обученные веса старого CRS не используются. Основная память — Qwen и копии её слоёв. fp16 на CUDA, GradScaler, checkpointing; на CPU — fp32. Если T4 не хватает, `--batch-size 4 --accumulation 32`. Фактические VRAM и скорость записываются после GPU-прогона; полного нового обучения здесь пока не было.

Full softmax по всем 32 462 трекам, loss = среднее отрицательное log-probability целей, каждый ход весит одинаково. Треки из истории не запрещаются. Метрики — Recall/Hit/NDCG@10/100 по ходам. По умолчанию фиксированная случайная выборка 5000 val-ходов; `--val-turns 0` оценивает все подготовленные val-ходы. Лучший checkpoint выбирается по NDCG@100. Результаты — `runs/.../metrics.jsonl`, `best.pt`, `last.pt`. Сохраняются новые веса, optimizer, scaler и позиция для продолжения; исходная Qwen загружается заново.

Для сравнения независимо обучить `--mode both`, `--mode text`, `--mode history` в разных `--out`, сохраняя одинаковые features/data/seed. В text история отключена; в history запрос заменён постоянным текстом. `--encoder-layers 0` — отдельный lazy baseline. Сначала пилот, затем расширение выборки по замеренной скорости. Полный корпус не обещаем обучить за одну Colab-сессию.

## Проверка на Mac

CPU уже проверен; для короткого запуска:

```bash
python -m twotower.train --device cpu --steps 3 --batch-size 1 --accumulation 1 --val-turns 10 --eval-every 3 --out runs/mac_cpu
```

GPU Apple называется MPS. Добавлен явный выбор устройства; на CPU/MPS используется fp32. В этой среде `torch.backends.mps.is_available()` возвращает False, поэтому работоспособность GPU-пути здесь не подтверждена. Пробовать с маленьким batch; неподдержанные MPS-операции могут выполняться на CPU через официальный fallback:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m twotower.train --device mps --steps 3 --batch-size 1 --accumulation 1 --val-turns 10 --eval-every 3 --out runs/mac_mps
```

См. [MPS в PyTorch 2.8](https://docs.pytorch.org/docs/2.8/notes/mps.html) и [CPU fallback](https://docs.pytorch.org/docs/2.8/mps_environment_variables.html). `auto` по-прежнему выбирает CUDA либо CPU; MPS нужно указать явно.

Проверки без скачивания моделей: `python -m pytest tests/test_twotower.py` (нужен pytest). Используют маленькую случайную Qwen и проверяют время истории, маски, loss, метрики, обучение, сохранение и переключение признаков.

Локально уже собраны `features.npz` (~182 МиБ) и `pilot/` (~106 МиБ): 82 122 train-хода и 8 421 val-ход. Прошли 10 тестов, один шаг настоящей Qwen на CPU, загрузка checkpoint, оценка и просмотр рекомендаций. Это техническая проверка; качество модели и CUDA/fp16 на T4 ещё предстоит измерить. `runs/cpu_smoke/` и `data/model/smoke/` — только отладочные результаты, для основного эксперимента используем `pilot/` и новый каталог запуска.

## Диагностика готовых checkpoint на CPU

```bash
../recsys/.venv/bin/python -u -m twotower.diagnose \
  --runs runs/lora_v2_e2 --out runs/lora_v2_e2/diagnostics_cpu \
  --threads 4 --batch-size 16
```

Можно использовать обычный `python`, если установлены зависимости. Нужны локальные
`both/best.pt`, `text/best.pt`, `history/best.pt`, готовый pilot/features и кеш Qwen.
Скрипт не обращается к сети и не меняет checkpoint.

На всех 8421 val-ходах оцениваются both с настоящей/пустой/чужой историей,
text и history с настоящей/чужой историей. Доноры — перестановка между разными
пользователями, фиксированный seed=42. Подменяется вся история, включая прошлые ходы.
Сохраняются top-100, поэтому повторный запуск продолжает недосчитанные варианты.
Кеш привязан к SHA256 данных, признаков и checkpoint.

`report.md` и `summary.json` содержат метрики первых/последующих ходов, recall
целей из истории и новых целей, paired bootstrap по диалогам (2000 повторов).
Для seen/new усредняются ходы с непустым соответствующим подмножеством целей;
позиции исходной выдачи сохраняются. Разные диалоги одного пользователя не объединяются
в bootstrap, что ограничивает интерпретацию интервалов.
`examples.md` показывает конкретные выдачи, `behaviour.json` — разнообразие top-1
и чувствительность к чужой истории. `history_gradients.json` — проверка градиентов
и одного малого SGD-обновления в RAM без сохранения изменённых весов.
CPU использует FP32, поэтому возможны небольшие отличия от FP16-оценки Colab.

## Следующие проверки: длина истории и каналы

Локальная оценка уже обученных моделей both/history на последних 16/32/64 треках:

```bash
../recsys/.venv/bin/python -u -m twotower.history_sweep
```

Результаты: `runs/lora_v2_e2/history_sweep/report.md` и JSON. На всех 8421 ходах.
Веса обучались с историей 64: это усечение на инференсе, не обучение с другой длиной.
Группы seen/new фиксированы по исходной истории 64; отдельно указано покрытие целей
каждой длиной. Для history/64 также считаются точные ранги целей во всём каталоге,
Recall@1000/10000 и медиана ранга. Кеш привязан к hash данных и checkpoint.

Colab: [colab_ablations.ipynb](colab_ablations.ipynb), архив `runs/twotower_experiments.zip`.
Зависимости и данные те же. В последней ячейке выбираются EXPERIMENTS:

| Запуск | История | Каналы | Сравнение |
|---|---:|---|---|
| history32 | 32 | audio,text,tags | Только длина против старого both/64 |
| no_tags | 64 | audio,text | Без обучаемой таблицы тегов |
| no_audio | 64 | text,tags | Без MuQ |

Все — новые обучения both на четыре эпохи. Не продолжают старые веса.
Перед длинным запуском 20 шагов каждого варианта дают оценку времени и сохраняются.
Если четыре эпохи выбранных вариантов не помещаются, скрипт останавливается;
можно выбрать один вариант. Бюджет не сокращается автоматически до двух эпох,
поскольку сравнение идёт с исходными четырьмя эпохами.

```bash
python -u -m twotower.ablations --jobs history32 --out /path/on/drive/runs/ablations_v2_e2 --hours 22
python -u -m twotower.ablations --jobs no_tags no_audio --out /path/on/drive/runs/ablations_v2_e2 --hours 22
```

`train --history-limit 32` обрезает историю в collator одинаково для train и val,
не требует пересборки pilot. Таблица позиций сохраняет размер 64, поэтому абляция
длины не меняет число параметров и их начальную инициализацию. Ограничение сохраняется
в config/checkpoint и применяется при оценке; менять его при resume запрещено.
Истории длиннее 64 в текущем pilot нет: для 128 нужна подготовка новых данных.
