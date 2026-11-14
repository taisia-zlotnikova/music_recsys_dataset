"""На готовых весах: история 16/32/64 и точные ранги целей history-only во всём каталоге."""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import Collator, TurnDataset
from .diagnose import aggregates, infer, per_turn_metrics, write_json
from .features import file_hash
from .model import build_model, load_trainable


def target_ranks(scores, targets):
    """Ранг от 1; при равных score меньший индекс трека идёт раньше."""
    catalog = torch.arange(scores.shape[1], device=scores.device)
    answer = []
    for row, ids in zip(scores, targets):
        ids = ids[ids >= 0]
        selected = row[ids, None]
        ranks = (row[None, :] > selected).sum(1) + ((row[None, :] == selected) & (catalog[None, :] < ids[:, None])).sum(1) + 1
        answer.append(ranks.tolist())
    return answer


def deep_ranks(model, parts, rows, collator, path, batch_size):
    ranks = json.loads(path.read_text()) if path.exists() else [None] * len(rows)
    pending = [i for i, r in enumerate(ranks) if r is None]
    started = last = time.monotonic()
    for offset in range(0, len(pending), batch_size):
        indices = pending[offset:offset + batch_size]
        batch = collator([rows[i] for i in indices])
        with torch.inference_mode():
            scores = model.scores(batch, parts)
            if not torch.isfinite(scores).all():
                raise FloatingPointError('Нечисловые scores')
            values = target_ranks(scores, batch['targets'])
        for i, r in zip(indices, values):
            ranks[i] = r
        if time.monotonic() - last >= 30 or offset + batch_size >= len(pending):
            write_json(path, ranks)
            print(f'history64 ranks: {len(rows)-len(pending)+min(offset+batch_size,len(pending))}/{len(rows)}', flush=True)
            last = time.monotonic()
    flat = np.concatenate([np.asarray(r) for r in ranks])
    return {'recall': {f'@{k}': float(np.mean([np.mean(np.asarray(r) <= k) for r in ranks]))
                       for k in (10, 100, 1000, 10000)},
            'median_target_rank_micro': float(np.median(flat)),
            'target_rank_quantiles_micro': np.quantile(flat, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
            'target_occurrences': len(flat),
            'share_top100_targets_in_top10_micro': float((flat <= 10).sum() / max(1, (flat <= 100).sum()))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', default='runs/lora_v2_e2')
    p.add_argument('--data', default='data/model/pilot')
    p.add_argument('--features', default='data/model/features.npz')
    p.add_argument('--out', default='runs/lora_v2_e2/history_sweep')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    root = Path(a.data); manifest = json.loads((root/'manifest.json').read_text())
    dataset = TurnDataset(root/'val'); rows = [dataset[i] for i in range(len(dataset))]
    info = [json.loads(l) for s in dataset.manifest['shards'] for l in (root/'val'/s['path']/'examples.jsonl').read_text().splitlines()]
    provenance = {'data': file_hash(root/'manifest.json'), 'features': file_hash(a.features),
                  'weights': {m: file_hash(Path(a.runs)/m/'best.pt') for m in ('both','history')}, 'schema': 1}
    if (out/'provenance.json').exists() and json.loads((out/'provenance.json').read_text()) != provenance:
        raise ValueError('Кеш от других данных или весов; выбери новую out')
    write_json(out/'provenance.json', provenance)
    with np.load(a.features) as pack:
        features = {k: pack[k] for k in pack.files}
    results = {}
    for mode in ('both','history'):
        saved = torch.load(Path(a.runs)/mode/'best.pt', map_location='cpu', weights_only=False, mmap=True)
        config = saved['config']
        assert saved['data_sha256'] == provenance['data'] and config['feature_sha256'] == provenance['features']
        assert config['history_len'] == 64 and config.get('history_limit', 64) == 64
        model = build_model(features, config, torch.device('cpu'), True)
        load_trainable(model, saved['weights']); del saved
        model.eval()
        with torch.inference_mode():
            parts = model.item_parts()
        for length in (16,32,64):
            short = [dict(row, history=row['history'][-length:]) for row in rows]
            name = f'{mode}_{length}'
            # Только кеш с проверенными hash данных/весов; не пересчитываем уже готовый baseline 64.
            previous = Path(a.runs)/'diagnostics_cpu'
            cache = previous/f'{mode}_real.npy'
            old = json.loads((previous/'provenance.json').read_text()) if (previous/'provenance.json').exists() else {}
            if (length == 64 and cache.exists() and old.get('data_sha256') == provenance['data']
                    and old.get('feature_sha256') == provenance['features']
                    and old.get('checkpoints',{}).get(mode) == provenance['weights'][mode]
                    and np.load(cache).shape == (len(rows),100) and np.all(np.load(cache) >= 0)):
                top = np.load(cache)
            else:
                top = infer(model, parts, short, Collator(manifest['pad_id'], manifest['history_only_tokens'], mode),
                            np.arange(len(rows)), 'real', out/f'{name}.npy', a.batch_size)
            # Группы целей фиксируются по исходной истории 64, чтобы сравнивать одинаковые цели.
            results[name] = {'metrics': aggregates(per_turn_metrics(top, rows), info),
                             'target_coverage': float(np.mean([np.isin(r['targets'],r['history'][:,0]).mean() for r in short])),
                             'mean_history_length': float(np.mean([len(r['history']) for r in short]))}
            write_json(out/'metrics.json', results)
        if mode == 'history':
            result = deep_ranks(model, parts, rows, Collator(manifest['pad_id'],manifest['history_only_tokens'],mode),
                                out/'history64_target_ranks.json', a.batch_size)
            write_json(out/'history64_rank_summary.json', result)
        del model, parts; gc.collect()
    lines = ['# История 16/32/64: инференс готовых моделей, обученных с лимитом 64', '',
             'Все 8421 val-хода, CPU FP32. Это проверка усечения входа, не сравнение новых обучений.', '',
             '| Модель | История | Recall@10 | Recall@100 | NDCG@100 | Покрытие целей историей |', '|---|---:|---:|---:|---:|---:|']
    for name, r in results.items():
        mode, length = name.split('_'); m = r['metrics']['all']
        lines.append(f'| {mode} | {length} | {m["recall@10"]["value"]:.4f} | {m["recall@100"]["value"]:.4f} | {m["ndcg@100"]["value"]:.4f} | {r["target_coverage"]:.4f} |')
    lines += ['', 'Seen/new в metrics.json всегда определяются по исходной истории 64.', '',
              '## Глубина поиска history-only/64', '', 'Recall — средний по ходам, ранги/квантили — по вхождениям целей. При равных score порядок по индексу трека.', '',
              '```json', json.dumps(result,indent=2), '```']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'Готово: {out / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
