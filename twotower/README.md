# Двухбашенная модель

Запускать из корня репозитория, Python 3.10+. Первое обучение — замороженная Qwen и обучаемые новые части.

```bash
pip install -r twotower/requirements.txt
python -m twotower.features --recsys ../recsys
python -m twotower.prepare
python -m twotower.show --index 0
```

`features` собирает все 32 462 ID из `reports/tracks.csv`: готовые MuQ-128, Qwen-1024 («исполнитель — название»), ID тегов и веса. Пересчёта нейросетевых признаков нет. Теги/жанры нормализуются; редкие (<5 треков) исключаются, остаются до 32 на трек. Веса Last.fm /100 и вес жанра 0.5, затем нормализация суммы. Общий файл `data/model/features.npz` можно отдать Паше. Внутри есть `track_ids`, `audio`, `text`, `tag_vocab`, `tag_ids`, `tag_weights`, `artists`, `titles`; строка в массивах всегда соответствует `track_ids`.

Если `reports/tracks.csv` ещё нет, его создаёт `python3 scripts/stats.py`. Для обучения достаточно готового `features.npz`: исходные parquet и соседний проект нужны только для его сборки.

## Выбор признаков трека

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
```

После установки, если Colab попросит, перезапустить сессию Python и снова выполнить mount/cd. Проверочный запуск с сохранением на Drive:

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
