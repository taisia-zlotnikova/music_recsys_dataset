"""Посмотреть подготовленный ход, его историю/цели и, при наличии checkpoint, рекомендации."""
import argparse
import bisect
import json
from pathlib import Path

import numpy as np
import torch

from .data import Collator, TurnDataset
from .features import check_catalog, file_hash
from .model import build_model, load_trainable
from .train import autocast, select_device, to_device


def example_info(dataset, index):
    shard = bisect.bisect_right(dataset.ends, index)
    local = index - (dataset.ends[shard - 1] if shard else 0)
    path = dataset.root / dataset.manifest['shards'][shard]['path'] / 'examples.jsonl'
    with path.open(encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == local:
                return json.loads(line)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/model/pilot')
    p.add_argument('--features', default='data/model/features.npz')
    p.add_argument('--split', choices=['train', 'val'], default='val')
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--checkpoint')
    p.add_argument('--top-k', type=int, default=10)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    p.add_argument('--local-files-only', action='store_true')
    a = p.parse_args()
    root = Path(a.data)
    manifest = json.loads((root / 'manifest.json').read_text())
    dataset = TurnDataset(root / a.split)
    if not 0 <= a.index < len(dataset) or a.top_k < 1:
        p.error('Нужен существующий индекс и положительный top-k')
    check_catalog(root, a.features, manifest)
    row, info = dataset[a.index], example_info(dataset, a.index)
    with np.load(a.features) as features:
        def track(index):
            return f"{features['track_ids'][index]}  {features['artists'][index]} — {features['titles'][index]}"
        print(f"Диалог {info['dialog_id']}, ход {info['turn']}, пользователь {info['user_id']}")
        print('\nЗапрос и контекст:\n' + info['text'])
        print(f"\nИстория ({len(row['history'])} уникальных треков, показаны последние 10):")
        for index, count, age in row['history'][-10:]:
            print(f'  {track(index)} [count_bucket={count}, age_bucket={age}]')
        print('\nЦелевые треки:')
        for index in row['targets']:
            print('  ' + track(index))
        if a.checkpoint:
            saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
            config = saved['config']
            if config['feature_sha256'] != file_hash(a.features):
                raise ValueError('Checkpoint обучен на другом файле признаков')
            device = select_device(a.device)
            model = build_model(features, config, device, a.local_files_only)
            load_trainable(model, saved['weights'])
            model.eval()
            batch = Collator(manifest['pad_id'], manifest['history_only_tokens'], config['mode'])([row])
            with torch.no_grad(), autocast(device):
                scores = model.scores(to_device(batch, device))[0]
            print('\nРекомендации:')
            for index in scores.topk(min(a.top_k, len(scores))).indices.tolist():
                marker = ' *' if index in row['targets'] else ''
                print('  ' + track(index) + marker)


if __name__ == '__main__':
    main()
