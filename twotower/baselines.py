"""Популярность и recency на том же pilot/val и с теми же метриками, что у моделей."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .data import TurnDataset
from .features import check_catalog, file_hash
from .model import retrieval_metrics


def recent_ranking(history, popular, k):
    recent = list(dict.fromkeys(int(t) for t in reversed(history)))
    seen = set(recent)
    return (recent + [int(t) for t in popular[:k + len(recent)] if int(t) not in seen])[:k]


def evaluate_baselines(data, features, val_turns=5000, seed=42):
    root = Path(data)
    manifest = json.loads((root / 'manifest.json').read_text())
    check_catalog(root, features, manifest)
    with np.load(features) as f:
        n_tracks = len(f['track_ids'])
    train, val = TurnDataset(root / 'train'), TurnDataset(root / 'val')
    counts = np.zeros(n_tracks, dtype=np.int64)
    for i in range(len(train)):
        np.add.at(counts, train[i]['targets'], 1)
    popular = np.argsort(-counts, kind='stable')
    indices = np.arange(len(val), dtype=np.int64)
    if val_turns and val_turns < len(val):
        indices = np.sort(np.random.default_rng(seed).choice(len(val), val_turns, replace=False))
    totals = {name: {} for name in ('popular', 'recent_then_popular')}
    coverage, empty, K = 0.0, 0, min(100, n_tracks)
    for start in range(0, len(indices), 128):
        rows = [val[int(i)] for i in indices[start:start + 128]]
        targets = torch.full((len(rows), max(len(r['targets']) for r in rows)), -1, dtype=torch.long)
        for i, row in enumerate(rows):
            targets[i, :len(row['targets'])] = torch.tensor(row['targets'].copy())
            h = set(map(int, row['history'][:, 0]))
            empty += not h
            coverage += len(h.intersection(row['targets'])) / len(row['targets'])
        for name in totals:
            # Ранги кодируем уникальными score: вычисление метрик общее с нейромоделью.
            scores = torch.zeros(len(rows), n_tracks)
            for i, row in enumerate(rows):
                ranking = popular[:K] if name == 'popular' else recent_ranking(row['history'][:, 0], popular, K)
                scores[i, torch.tensor(ranking, dtype=torch.long)] = torch.arange(K, 0, -1, dtype=torch.float32)
            for key, value in retrieval_metrics(scores, targets).items():
                totals[name][key] = totals[name].get(key, 0) + value
    return {'data_sha256': file_hash(root / 'manifest.json'), 'feature_sha256': file_hash(features),
            'train_turns': len(train), 'val_turns': len(indices), 'seed': seed,
            'val_indices_sha256': hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
            'empty_histories': empty, 'target_coverage_in_history': coverage / len(indices),
            'metrics': {name: {k: v / len(indices) for k, v in values.items()} for name, values in totals.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/model/pilot')
    p.add_argument('--features', default='data/model/features.npz')
    p.add_argument('--val-turns', type=int, default=5000, help='0 = все подготовленные val-ходы')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default='runs/baselines.json')
    a = p.parse_args()
    if a.val_turns < 0:
        p.error('--val-turns должен быть неотрицательным')
    torch.set_num_threads(2)
    result = evaluate_baselines(a.data, a.features, a.val_turns, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
