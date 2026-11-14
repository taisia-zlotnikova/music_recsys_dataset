"""Компактные шарды ходов; загрузка через mmap, без всего корпуса в RAM."""
import bisect
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

INSTRUCTION = 'Retrieve music tracks that match the request and listening history.'


def model_path(name, local_files_only=False):
    # В transformers 4.57.6 AutoTokenizer иногда делает model_info даже с local_files_only.
    # Передача пути к snapshot гарантирует работу с уже скачанными файлами без сети.
    if local_files_only and not Path(name).is_dir():
        from huggingface_hub import snapshot_download
        return snapshot_download(name, local_files_only=True)
    return name


class TextEncoder:
    def __init__(self, tokenizer, max_tokens=128):
        self.tokenizer = tokenizer
        self.prefix = tokenizer.encode(f'Instruct: {INSTRUCTION}\nQuery:\n', add_special_tokens=False)
        self.eos = tokenizer.eos_token_id
        self.pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.eos
        self.budget = max_tokens - len(self.prefix) - 1
        if self.eos is None or self.budget < 1:
            raise ValueError('Нужны EOS и место для текста запроса')
        self.history_only = self.prefix + tokenizer.encode('Recommend music.', add_special_tokens=False) + [self.eos]

    def encode(self, texts):
        tokens = self.tokenizer(texts, add_special_tokens=False)['input_ids']
        # Сохраняем конец: текущую реплику и ближайший к ней контекст.
        return [self.prefix + t[-self.budget:] + [self.eos] for t in tokens]


class ShardWriter:
    def __init__(self, root, shard_size=5000):
        self.root, self.shard_size = Path(root), shard_size
        self.pending, self.shards, self.count = [], [], 0

    def add(self, example):
        self.pending.append(example)
        if len(self.pending) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        directory = self.root / f'{len(self.shards):05d}'
        directory.mkdir(parents=True)
        for key in ['tokens', 'history', 'targets']:
            values = [np.asarray(r[key], dtype=np.int32).reshape(-1, 3) if key == 'history'
                      else np.asarray(r[key], dtype=np.int32) for r in self.pending]
            offsets = np.r_[0, np.cumsum([len(v) for v in values])]
            np.save(directory / f'{key}.npy', np.concatenate(values))
            np.save(directory / f'{key}_offsets.npy', offsets)
        with (directory / 'examples.jsonl').open('w', encoding='utf-8') as f:
            for row in self.pending:
                f.write(json.dumps({k: v for k, v in row.items() if k not in ('tokens', 'history')},
                                   ensure_ascii=False) + '\n')
        self.count += len(self.pending)
        self.shards.append({'path': directory.name, 'examples': len(self.pending)})
        self.pending.clear()

    def finish(self):
        self.flush()
        manifest = {'examples': self.count, 'shards': self.shards}
        (self.root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        return manifest


class TurnDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        self.ends = np.cumsum([s['examples'] for s in self.manifest['shards']]).tolist()
        self.cache = OrderedDict()

    def __len__(self):
        return self.manifest['examples']

    def __getitem__(self, index):
        shard = bisect.bisect_right(self.ends, index)
        local = index - (self.ends[shard - 1] if shard else 0)
        if shard not in self.cache:
            directory = self.root / self.manifest['shards'][shard]['path']
            self.cache[shard] = {k: np.load(directory / f'{k}.npy', mmap_mode='r')
                                 for key in ('tokens', 'history', 'targets')
                                 for k in (key, key + '_offsets')}
            if len(self.cache) > 4:
                self.cache.popitem(last=False)
        self.cache.move_to_end(shard)
        arrays = self.cache[shard]
        return {key: arrays[key][arrays[key + '_offsets'][local]:arrays[key + '_offsets'][local + 1]]
                for key in ('tokens', 'history', 'targets')}


class ShardSampler(Sampler):
    """Перемешиваем шарды и строки внутри: память не зависит от размера корпуса."""
    def __init__(self, dataset, seed):
        self.dataset, self.seed, self.epoch = dataset, seed, 0

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        starts = [0] + self.dataset.ends[:-1]
        for shard in rng.permutation(len(starts)):
            for i in rng.permutation(self.dataset.ends[shard] - starts[shard]):
                yield starts[shard] + int(i)


class Collator:
    def __init__(self, pad_id, history_only_tokens, mode='both', history_limit=None):
        self.pad, self.history_only, self.mode = pad_id, history_only_tokens, mode
        if history_limit is not None and history_limit < 1:
            raise ValueError('history_limit должен быть положительным')
        self.history_limit = history_limit

    def __call__(self, rows):
        histories = [r['history'] if self.mode != 'text' else np.empty((0, 3), np.int32) for r in rows]
        if self.history_limit is not None:
            histories = [h[-self.history_limit:] for h in histories]
        texts = [r['tokens'] if self.mode != 'history' else self.history_only for r in rows]
        B, H, T, P = len(rows), max(map(len, histories)), max(map(len, texts)), max(len(r['targets']) for r in rows)
        b = {'hist': torch.zeros(B, H, dtype=torch.long), 'hcount': torch.zeros(B, H, dtype=torch.long),
             'hage': torch.zeros(B, H, dtype=torch.long), 'hmask': torch.zeros(B, H, dtype=torch.bool),
             'tok': torch.full((B, T), self.pad, dtype=torch.long), 'mask': torch.zeros(B, T, dtype=torch.bool),
             'targets': torch.full((B, P), -1, dtype=torch.long)}
        for i, (history, tokens, row) in enumerate(zip(histories, texts, rows)):
            if len(history):
                h = torch.tensor(np.array(history), dtype=torch.long)
                b['hist'][i, -len(h):], b['hcount'][i, -len(h):], b['hage'][i, -len(h):] = h.T
                b['hmask'][i, -len(h):] = True
            b['tok'][i, -len(tokens):] = torch.tensor(np.array(tokens), dtype=torch.long)
            b['mask'][i, -len(tokens):] = True
            b['targets'][i, :len(row['targets'])] = torch.tensor(np.array(row['targets']), dtype=torch.long)
        return b
