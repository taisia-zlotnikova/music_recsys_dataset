"""Выбрать диалоги, присоединить строго прошлое Onion и сохранить ходы для обучения."""
import argparse
import gzip
import json
import math
import random
import sqlite3
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from .data import ShardWriter, TextEncoder, model_path
from .features import file_hash

MODEL = 'Qwen/Qwen3-Embedding-0.6B'


def select_dialogues(path, count, seed, scan_limit=0):
    """Reservoir: равномерная выборка целых диалогов, независимо от порядка строк."""
    rng, selected, seen = random.Random(seed), [], 0
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if scan_limit and line_no > scan_limit:
                break
            if line_no % 5_000_000 == 0:
                print(f'{path}: просмотрено {line_no:,} строк', flush=True)
            row = json.loads(line)
            if row.get('clean_v2') is not True or not row.get('turns'):
                continue
            seen += 1
            item = (line_no, row)
            if not count or len(selected) < count:
                selected.append(item)
            else:
                j = rng.randrange(seen)
                if j < count:
                    selected[j] = item
    return selected, seen


def store_dialogues(db, paths, counts, seed, scan_limit):
    db.execute('CREATE TABLE dialogs (user INTEGER, start INTEGER, split TEXT, id TEXT, row TEXT)')
    stats = {}
    for i, (split, path) in enumerate(paths.items()):
        # При count=0 потоковый вариант нужен, чтобы все 50M диалогов не оказались в RAM.
        if counts[split] == 0:
            opener = gzip.open if str(path).endswith('.gz') else open
            selected, seen = 0, 0
            with opener(path, 'rt', encoding='utf-8') as f:
                batch = []
                for line_no, line in enumerate(f, 1):
                    if scan_limit and line_no > scan_limit:
                        break
                    if line_no % 5_000_000 == 0:
                        print(f'{path}: просмотрено {line_no:,} строк', flush=True)
                    row = json.loads(line)
                    if row.get('clean_v2') is not True or not row.get('turns'):
                        continue
                    batch.append((int(row['user_id']), int(row['ts_start']), split,
                                  f'{split}:{line_no}', line))
                    selected += 1
                    seen += 1
                    if len(batch) >= 5000:
                        db.executemany('INSERT INTO dialogs VALUES (?,?,?,?,?)', batch)
                        batch.clear()
                db.executemany('INSERT INTO dialogs VALUES (?,?,?,?,?)', batch)
        else:
            rows, seen = select_dialogues(path, counts[split], seed + i, scan_limit)
            db.executemany('INSERT INTO dialogs VALUES (?,?,?,?,?)', [
                (int(r['user_id']), int(r['ts_start']), split, f'{split}:{line}', json.dumps(r))
                for line, r in rows])
            selected = len(rows)
        db.commit()
        stats[split] = {'seen_dialogues': seen, 'selected_dialogues': selected}
        print(f'{split}: выбрано {selected:,} из {seen:,} диалогов', flush=True)
    db.execute('CREATE INDEX dialog_order ON dialogs(user,start)')
    return stats


def collect_events(db, path, track_pos):
    """Один проход по Onion; выбранные пользователи и каталог, сортировка на диске."""
    db.execute('CREATE TABLE events (user INTEGER, ts INTEGER, track INTEGER)')
    ends = dict(db.execute('SELECT user,MAX(start) FROM dialogs GROUP BY user'))
    users = pa.array(list(ends), type=pa.int64())
    cutoff_map = ends
    seen, kept, started = 0, 0, time.monotonic()
    path = str(path)
    compression = 'bz2' if path.endswith('.bz2') else 'gzip' if path.endswith('.gz') else None
    with pa.input_stream(path, compression=compression) as stream:
        reader = pacsv.open_csv(stream, read_options=pacsv.ReadOptions(block_size=8 << 20),
            parse_options=pacsv.ParseOptions(delimiter='\t'),
            convert_options=pacsv.ConvertOptions(column_types={
                'user_id': pa.int64(), 'track_id': pa.string(), 'timestamp': pa.timestamp('s')}))
        next_log = 5_000_000
        for block in reader:
            seen += len(block)
            block = block.filter(pc.is_in(block.column('user_id'), value_set=users))
            user = block.column('user_id').to_pylist()
            tracks = block.column('track_id').to_pylist()
            timestamps = block.column('timestamp').cast(pa.int64()).to_pylist()
            events = [(u, ts, track_pos[t]) for u, t, ts in zip(user, tracks, timestamps)
                      if t in track_pos and ts < cutoff_map[u]]
            db.executemany('INSERT INTO events VALUES (?,?,?)', events)
            kept += len(events)
            if seen >= next_log:
                db.commit()
                print(f'Onion: {seen:,} строк, сохранено {kept:,}, {(time.monotonic()-started)/60:.1f} мин', flush=True)
                next_log += 5_000_000
    db.commit()
    print('Индекс истории на диске…', flush=True)
    db.execute('CREATE INDEX event_order ON events(user,ts,track)')
    db.commit()
    return {'raw_events_scanned': seen, 'events_kept': kept, 'users': len(ends)}


def bucket(value):
    return min(15, int(math.log2(max(1, value))))


def history_array(state, prefix, cutoff, length):
    """state: track → [число реальных plays, последний ts]. Prefix известен только внутри диалога."""
    added = {}
    for order, track in enumerate(prefix):
        n, _ = added.get(track, (0, 0))
        added[track] = (n + 1, order)
    recent = sorted(state, key=lambda t: (state[t][1], t))[-length:]
    recent = [t for t in recent if t not in added] + sorted(added, key=lambda t: added[t][1])
    result = []
    for t in recent[-length:]:
        count, last = state.get(t, (0, cutoff))
        count += added.get(t, (0, 0))[0]
        age = 0 if t in added else bucket((cutoff - last) / 86400 + 1)
        result.append([t, bucket(count), age])
    return result


def dialogue_turns(row, dialog_id, state, track_pos, history_len):
    tracks = [track_pos[t] for t in row['tracks']]
    previous, text, offset = [], [], 0
    for turn, (lines, k) in enumerate(row['turns']):
        if not isinstance(k, int) or k < 1 or not lines or not all(isinstance(s, str) and s.strip() for s in lines):
            raise ValueError(f'{dialog_id}, ход {turn}: некорректный ход')
        targets = list(dict.fromkeys(tracks[offset:offset + k]))
        if offset + k > len(tracks):
            raise ValueError(f'{dialog_id}: k выходит за tracks')
        text.extend('User: ' + line.strip() for line in lines)
        yield {'dialog_id': dialog_id, 'turn': turn, 'user_id': int(row['user_id']),
               'session_start': int(row['ts_start']), 'text': '\n'.join(text), 'targets': targets,
               'history': history_array(state, previous, row['ts_start'], history_len)}
        previous.extend(tracks[offset:offset + k])
        offset += k
    if offset != len(tracks):
        raise ValueError(f'{dialog_id}: сумма k не равна числу tracks')


def write_turns(db, track_pos, encoder, root, history_len, shard_size=5000):
    writers = {s: ShardWriter(Path(root) / s, shard_size) for s in ('train', 'val')}
    pending = []
    def flush():
        tokens = encoder.encode([row['text'] for _, row in pending])
        for (split, row), tok in zip(pending, tokens):
            row['tokens'] = tok
            writers[split].add(row)
        pending.clear()
    for (user,) in db.execute('SELECT DISTINCT user FROM dialogs ORDER BY user'):
        events = iter(db.execute('SELECT ts,track FROM events WHERE user=? ORDER BY ts,track', (user,)))
        event, state = next(events, None), {}
        for cutoff, split, dialog_id, raw in db.execute(
                'SELECT start,split,id,row FROM dialogs WHERE user=? ORDER BY start,id', (user,)):
            while event is not None and event[0] < cutoff:
                ts, track = event
                count, _ = state.get(track, (0, 0))
                state[track] = (count + 1, ts)
                event = next(events, None)
            for row in dialogue_turns(json.loads(raw), dialog_id, state, track_pos, history_len):
                pending.append((split, row))
                if len(pending) >= 128:
                    flush()
    if pending:
        flush()
    return {split: writer.finish() for split, writer in writers.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', default='data/model/features.npz')
    p.add_argument('--train', default='data/split_v2/train.jsonl.gz')
    p.add_argument('--val', default='data/split_v2/val.jsonl.gz')
    p.add_argument('--onion', default='_archive/userid_trackid_timestamp.tsv.bz2')
    p.add_argument('--out', default='data/model/pilot')
    p.add_argument('--train-dialogs', type=int, default=50000, help='0 = все')
    p.add_argument('--val-dialogs', type=int, default=5000, help='0 = все')
    p.add_argument('--history-len', type=int, default=64)
    p.add_argument('--max-tokens', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--model', default=MODEL)
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--scan-limit', type=int, default=0, help='Только отладка: первые N строк каждого split, выборка смещена')
    a = p.parse_args()
    if min(a.train_dialogs, a.val_dialogs, a.scan_limit) < 0 or a.history_len < 1:
        p.error('Размеры должны быть неотрицательными, история — положительной')
    root = Path(a.out)
    if root.exists():
        raise FileExistsError(root)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path(a.model, a.local_files_only),
                                             local_files_only=a.local_files_only)
    encoder = TextEncoder(tokenizer, a.max_tokens)
    root.mkdir(parents=True, exist_ok=False)
    with np.load(a.features) as f:
        track_ids = f['track_ids']
        track_pos = {t: i for i, t in enumerate(track_ids)}
    np.save(root / 'track_ids.npy', track_ids)
    with sqlite3.connect(root / 'work.sqlite') as db:
        db.execute('PRAGMA journal_mode=OFF')
        db.execute('PRAGMA synchronous=OFF')
        selected = store_dialogues(db, {'train': a.train, 'val': a.val},
                                  {'train': a.train_dialogs, 'val': a.val_dialogs}, a.seed, a.scan_limit)
        if any(s['selected_dialogues'] == 0 for s in selected.values()):
            raise ValueError('Нужны непустые train и val')
        events = collect_events(db, a.onion, track_pos)
        splits = write_turns(db, track_pos, encoder, root, a.history_len)
    # Удаляется только созданная этим запуском временная база, не исходные данные.
    (root / 'work.sqlite').unlink()
    manifest = {'model': a.model, 'feature_sha256': file_hash(a.features), 'seed': a.seed,
                'history_len': a.history_len, 'max_tokens': a.max_tokens, 'pad_id': encoder.pad,
                'history_only_tokens': encoder.history_only, 'debug_scan_limit': a.scan_limit,
                'selection': selected, 'events': events, 'splits': splits,
                'history_rule': 'Onion timestamp < session_start; then previous turns only'}
    (root / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({s: m['examples'] for s, m in splits.items()}, indent=2))


if __name__ == '__main__':
    main()
