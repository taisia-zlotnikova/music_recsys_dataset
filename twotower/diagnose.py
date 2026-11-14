"""CPU-диагностика checkpoint: история, группы целей, paired bootstrap по диалогам.

Предсказания кешируются; повторный запуск продолжает незаконченные варианты.
Обучение checkpoint не продолжается. Проверка градиентов меняет лишь временную модель в RAM.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from .baselines import recent_ranking
from .data import Collator, TurnDataset
from .features import check_catalog, file_hash
from .model import build_model, load_trainable

VARIANTS = {'both': ('real', 'empty', 'other'), 'text': ('real',), 'history': ('real', 'other')}


def write_json(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def other_users(users, seed):
    """Перестановка без повторов доноров и без истории того же пользователя."""
    users = np.asarray(users)
    unique, counts = np.unique(users, return_counts=True)
    shift = int(counts.max())
    if shift * 2 > len(users):
        raise ValueError('Слишком много ходов одного пользователя для перестановки без совпадений')
    rng = np.random.default_rng(seed)
    order = np.concatenate([rng.permutation(np.flatnonzero(users == u)) for u in rng.permutation(unique)])
    donors = np.empty(len(users), dtype=np.int64)
    donors[order] = np.roll(order, shift)
    assert np.all(users != users[donors]) and len(np.unique(donors)) == len(users)
    return donors


def per_turn_metrics(predictions, rows):
    """Seen/new: recall по соответствующим целям, позиции исходной выдачи сохранены."""
    n = len(rows)
    values = {f'{metric}@{k}': np.zeros(n) for k in (10, 100) for metric in ('recall', 'hit', 'ndcg')}
    for kind in ('seen', 'new'):
        for k in (10, 100):
            values[f'{kind}_recall@{k}'] = np.full(n, np.nan)
    for i, row in enumerate(rows):
        target = np.unique(row['targets'])
        seen = np.isin(target, row['history'][:, 0])
        for k in (10, 100):
            top = predictions[i, :k]
            hit = np.isin(top, target)
            values[f'recall@{k}'][i] = hit.sum() / len(target)
            values[f'hit@{k}'][i] = hit.any()
            discounts = 1 / np.log2(np.arange(len(top)) + 2)
            values[f'ndcg@{k}'][i] = np.dot(hit, discounts) / discounts[:min(len(target), len(top))].sum()
            for kind, mask in [('seen', seen), ('new', ~seen)]:
                if mask.any():
                    values[f'{kind}_recall@{k}'][i] = np.isin(target[mask], top).mean()
    return values


def aggregates(values, info):
    first = np.array([r['turn'] == 0 for r in info])
    result = {}
    for name, mask in [('all', np.ones(len(info), bool)), ('first', first), ('later', ~first)]:
        metrics = {}
        for key, array in values.items():
            eligible = mask & np.isfinite(array)
            metrics[key] = {'value': float(array[eligible].mean()) if eligible.any() else None,
                            'turns': int(eligible.sum())}
        result[name] = metrics
    return result


def bootstrap_difference(left, right, dialog_ids, seed=42, repeats=2000):
    """Разность средних по ходам; целые диалоги ресэмплируются с возвращением."""
    _, inverse = np.unique(dialog_ids, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=left - right)
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats)
    for i in range(repeats):
        selected = rng.integers(len(counts), size=len(counts))
        samples[i] = sums[selected].sum() / counts[selected].sum()
    return {'difference': float((left - right).mean()),
            'ci95': np.quantile(samples, [0.025, 0.975]).tolist(), 'dialogues': len(counts),
            'resamples': repeats, 'unit': 'dialogue; turn-weighted metric'}


def gradient_check(model, rows, collator):
    """Один диагностический SGD-шаг в памяти; файлы checkpoint не изменяются."""
    model.train()
    model.zero_grad(set_to_none=True)
    loss = model(collator(rows))
    loss.backward()
    groups = {}
    representatives = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group = ('tags' if name == 'item.tags.weight' else 'encoder_lora' if name.startswith('user.encoder.')
                 and 'lora_' in name else 'cross_lora' if name.startswith('user.cross.') and 'lora_' in name
                 else 'decoder_lora' if name.startswith('user.base.') and 'lora_' in name
                 else 'gates' if name.endswith('.gate') else 'history_input' if name.startswith('user.track_in.')
                 else 'other')
        stat = groups.setdefault(group, {'squared_norm': 0.0, 'parameters_with_grad': 0, 'finite': True})
        if param.grad is not None:
            norm = param.grad.detach().float().norm().item()
            stat['squared_norm'] += norm * norm
            stat['finite'] &= bool(torch.isfinite(param.grad).all())
            stat['parameters_with_grad'] += 1
            if norm > 0 and group not in representatives:
                representatives[group] = (name, param, param.detach().clone())
    frozen_grads = [n for n, p in model.named_parameters() if not p.requires_grad and p.grad is not None]
    finite = bool(torch.isfinite(loss)) and all(s['finite'] for s in groups.values())
    if finite:
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    p.add_(p.grad, alpha=-1e-4)
    for group, stat in groups.items():
        stat['gradient_norm'] = stat.pop('squared_norm') ** 0.5
        if group in representatives:
            name, param, old = representatives[group]
            stat['checked_parameter'] = name
            stat['max_update'] = float((param.detach() - old).abs().max())
    model.zero_grad(set_to_none=True)
    return {'loss': loss.item(), 'finite': finite, 'groups': groups, 'frozen_weights_with_grad': frozen_grads,
            'note': 'Один SGD-шаг lr=1e-4 с clip=1 только в RAM, не продолжение обучения. Checkpoint не перезаписан.'}


def infer(model, parts, rows, collator, donors, variant, path, batch_size):
    n = len(rows)
    if path.exists():
        top = np.load(path)
        if top.shape != (n, 100):
            raise ValueError(f'Неверная форма кеша {path}')
    else:
        top = np.full((n, 100), -1, dtype=np.int32)
    pending = np.flatnonzero(top[:, 0] < 0)
    # Меньше паддинга, при этом индексы результата всегда исходные.
    order = sorted(pending, key=lambda i: len(rows[i]['tokens']))
    started = time.monotonic()
    last_save = started
    for offset in range(0, len(order), batch_size):
        indices = order[offset:offset + batch_size]
        batch_rows = []
        for i in indices:
            row = dict(rows[i])
            if variant == 'empty':
                row['history'] = np.empty((0, 3), dtype=np.int32)
            elif variant == 'other':
                row['history'] = rows[donors[i]]['history']
            batch_rows.append(row)
        with torch.inference_mode():
            scores = model.scores(collator(batch_rows), parts)
            if not bool(torch.isfinite(scores).all()):
                raise FloatingPointError(path.name)
            top[indices] = scores.topk(100, dim=1).indices.numpy().astype(np.int32)
        if time.monotonic() - last_save >= 30 or offset + batch_size >= len(order):
            temporary = path.with_suffix('.partial')
            with temporary.open('wb') as f:
                np.save(f, top)
            temporary.replace(path)
            done = min(offset + batch_size, len(order))
            elapsed = time.monotonic() - started
            print(f'{path.stem}: {n - len(order) + done}/{n}; {done / max(elapsed, 1e-6):.1f} ходов/с; '
                  f'осталось ~{(len(order) - done) * elapsed / max(done, 1) / 60:.1f} мин', flush=True)
            last_save = time.monotonic()
    return top


def report(out, metrics, comparisons, gradients, predictions, rows, info, features, provenance):
    summary = {'provenance': provenance, 'metrics': metrics, 'paired_bootstrap': comparisons,
               'history_gradient_check': gradients}
    write_json(out / 'summary.json', summary)
    lines = ['# Диагностика encoder–decoder v2 на CPU', '',
             f'Полный pilot/val: {len(rows)} ходов. CPU FP32; исходный Colab использовал FP16.',
             'Файлы checkpoint не менялись. Bootstrap: 2000 повторов по целым диалогам; '
             'зависимость разных диалогов одного пользователя не учитывается.', '',
             '| Вариант | Recall@10 | Recall@100 | NDCG@100 |', '|---|---:|---:|---:|']
    for name, groups in metrics.items():
        m = groups['all']
        lines.append(f'| {name} | {m["recall@10"]["value"]:.4f} | {m["recall@100"]["value"]:.4f} | {m["ndcg@100"]["value"]:.4f} |')
    for group in ['first', 'later']:
        lines += ['', f'## {group}', '', '| Вариант | Ходов | Recall@100 | Seen recall@100 | New recall@100 |', '|---|---:|---:|---:|---:|']
        for name, groups in metrics.items():
            m = groups[group]
            lines.append(f'| {name} | {m["recall@100"]["turns"]} | {m["recall@100"]["value"]:.4f} | '
                         f'{m["seen_recall@100"]["value"]:.4f} | {m["new_recall@100"]["value"]:.4f} |')
    lines += ['', 'Seen/new — средний recall соответствующего подмножества целей по ходам, где оно непустое. '
              'Ход со смешанными целями входит в обе группы. Группы всегда определены по настоящей истории; '
              'позиции выдачи не меняются. Их средние нельзя просто сложить.', '', '## Разности и 95% CI', '']
    for name, measures in comparisons.items():
        for metric, value in measures.items():
            lines.append(f'- {name}, {metric}: {value["difference"]:.5f}, CI {value["ci95"]}.')
    lines += ['', '## Градиенты history-only', '', '```json', json.dumps(gradients, indent=2, ensure_ascii=False), '```',
              '', '## Ограничения', '',
              'Пустая/чужая история — вмешательство во вход готовой модели, не переобученная абляция. '
              'Переставляется вся история, включая прошлые ходы диалога; этот тест не изолирует только персонализацию. '
              'Перестановка одна, фиксированная, между разными пользователями. '
              'Ненулевые градиенты и gates не гарантируют хорошего использования истории.']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    # Примеры, где recency находит цель, а нейросетевая history-only нет, + контрольные примеры.
    recent, hist = predictions['recent_then_popular'], predictions['history_real']
    selected = [i for i in range(len(rows)) if np.isin(rows[i]['targets'], recent[i, :10]).any()
                and not np.isin(rows[i]['targets'], hist[i, :100]).any()][:8]
    selected += [i for i in range(min(4, len(rows))) if i not in selected]
    def track(t):
        return f'{features["artists"][t]} — {features["titles"][t]} ({features["track_ids"][t]})'
    examples = []
    for i in selected:
        row = rows[i]
        examples += [f'## {info[i]["dialog_id"]}, ход {info[i]["turn"]}, пользователь {info[i]["user_id"]}', '',
                     info[i]['text'], '', 'Цели: ' + '; '.join(track(t) for t in row['targets']), '',
                     'Последние 5 треков истории: ' + '; '.join(track(t) for t in row['history'][-5:, 0]), '']
        for name in ('both_real', 'both_empty', 'both_other', 'text_real', 'history_real', 'history_other', 'recent_then_popular'):
            examples += [f'**{name}**', '']
            examples += [f'- {track(t)}' + (' **ЦЕЛЬ**' if t in row['targets'] else '') for t in predictions[name][i, :5]]
            examples += ['']
    (out / 'examples.md').write_text('\n'.join(examples) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', default='runs/lora_v2_e2')
    p.add_argument('--data', default='data/model/pilot')
    p.add_argument('--features', default='data/model/features.npz')
    p.add_argument('--out', default='runs/lora_v2_e2/diagnostics_cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=16)
    a = p.parse_args()
    if min(a.threads, a.batch_size) < 1:
        p.error('threads и batch-size должны быть положительными')
    torch.set_num_threads(a.threads)
    torch.manual_seed(42)
    np.random.seed(42)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(a.data)
    manifest = json.loads((root / 'manifest.json').read_text())
    check_catalog(root, a.features, manifest)
    dataset = TurnDataset(root / 'val')
    rows = [dataset[i] for i in range(len(dataset))]
    info = []
    for shard in dataset.manifest['shards']:
        info.extend(json.loads(line) for line in (root / 'val' / shard['path'] / 'examples.jsonl').read_text().splitlines())
    assert len(rows) == len(info)
    donors = other_users([r['user_id'] for r in info], 42)
    provenance = {'data_sha256': file_hash(root / 'manifest.json'), 'feature_sha256': file_hash(a.features),
                  'checkpoints': {m: file_hash(Path(a.runs) / m / 'best.pt') for m in VARIANTS},
                  'seed': 42, 'device': 'cpu', 'dtype': 'float32', 'val_turns': len(rows), 'schema': 1}
    provenance_file = out / 'provenance.json'
    if provenance_file.exists() and json.loads(provenance_file.read_text()) != provenance:
        raise ValueError('Данные/веса не совпадают с кешем: выбери новую --out')
    write_json(provenance_file, provenance)
    np.save(out / 'history_donors.npy', donors)
    with np.load(a.features) as pack:
        features = {name: pack[name] for name in pack.files}
    predictions = {}
    for mode, variants in VARIANTS.items():
        needed = any(not (out / f'{mode}_{v}.npy').exists() or
                     (np.load(out / f'{mode}_{v}.npy')[:, 0] < 0).any() for v in variants)
        need_grad = mode == 'history' and not (out / 'history_gradients.json').exists()
        if needed or need_grad:
            saved = torch.load(Path(a.runs) / mode / 'best.pt', map_location='cpu', weights_only=False, mmap=True)
            config = saved['config']
            assert saved['data_sha256'] == provenance['data_sha256']
            assert config['feature_sha256'] == provenance['feature_sha256'] and config['mode'] == mode
            print(f'Загрузка {mode}, checkpoint step={saved["step"]}', flush=True)
            model = build_model(features, config, torch.device('cpu'), local_files_only=True)
            load_trainable(model, saved['weights'])
            del saved
            model.eval()
            collator = Collator(manifest['pad_id'], manifest['history_only_tokens'], mode)
            with torch.inference_mode():
                parts = model.item_parts()
            for variant in variants:
                name = f'{mode}_{variant}'
                predictions[name] = infer(model, parts, rows, collator, donors, variant, out / f'{name}.npy', a.batch_size)
            if need_grad:
                chosen = [0, len(rows) // 3, len(rows) // 2, len(rows) - 1]
                gradients = gradient_check(model, [rows[i] for i in chosen], collator)
                gradients['val_indices'] = chosen
                write_json(out / 'history_gradients.json', gradients)
            del model, parts
            gc.collect()
        else:
            for variant in variants:
                predictions[f'{mode}_{variant}'] = np.load(out / f'{mode}_{variant}.npy')
    train = TurnDataset(root / 'train')
    counts = np.zeros(len(features['track_ids']), dtype=np.int64)
    for i in range(len(train)):
        np.add.at(counts, train[i]['targets'], 1)
    popular = np.argsort(-counts, kind='stable')
    predictions['popular'] = np.tile(popular[:100], (len(rows), 1))
    predictions['recent_then_popular'] = np.array([recent_ranking(row['history'][:, 0], popular, 100) for row in rows])
    values = {name: per_turn_metrics(top, rows) for name, top in predictions.items()}
    metrics = {name: aggregates(v, info) for name, v in values.items()}
    comparisons = {}
    for left, right in [('both_real', 'recent_then_popular'), ('both_real', 'text_real'),
                        ('both_real', 'both_empty'), ('both_real', 'both_other'), ('history_real', 'history_other')]:
        comparisons[f'{left} - {right}'] = {metric: bootstrap_difference(values[left][metric], values[right][metric],
            [r['dialog_id'] for r in info]) for metric in ('recall@10', 'recall@100', 'ndcg@100')}
    for group in ('first', 'later'):
        selected = np.array([i for i, r in enumerate(info) if (r['turn'] == 0) == (group == 'first')])
        comparisons[f'both_real - recent_then_popular [{group}]'] = {
            metric: bootstrap_difference(values['both_real'][metric][selected],
                                         values['recent_then_popular'][metric][selected],
                                         [info[i]['dialog_id'] for i in selected])
            for metric in ('recall@100', 'ndcg@100')}
    gradients = json.loads((out / 'history_gradients.json').read_text())
    report(out, metrics, comparisons, gradients, predictions, rows, info, features, provenance)
    behaviour = {}
    for mode in ('both', 'history'):
        real, other = predictions[f'{mode}_real'], predictions[f'{mode}_other']
        _, frequencies = np.unique(real[:, 0], return_counts=True)
        behaviour[mode] = {'unique_top1': len(frequencies), 'most_common_top1_fraction': float(frequencies.max() / len(rows)),
                           'top1_changed_with_other_history': float((real[:, 0] != other[:, 0]).mean()),
                           'top10_overlap_with_other_history': float(np.mean([len(set(x[:10]) & set(y[:10])) / 10
                                                                              for x, y in zip(real, other)]))}
    write_json(out / 'behaviour.json', behaviour)
    reference_path = Path(a.runs) / 'results.json'
    if reference_path.exists():
        reference = json.loads(reference_path.read_text())
        reproduction = {mode: {key: {'cpu': metrics[f'{mode}_real']['all'][key]['value'],
                                     'colab': reference[mode]['metrics'][key],
                                     'difference': metrics[f'{mode}_real']['all'][key]['value'] - reference[mode]['metrics'][key]}
                               for key in ('recall@10', 'recall@100', 'ndcg@100')} for mode in VARIANTS}
        write_json(out / 'cpu_vs_colab.json', reproduction)
    assert all(file_hash(Path(a.runs) / mode / 'best.pt') == digest
               for mode, digest in provenance['checkpoints'].items()), 'Checkpoint изменился во время диагностики'
    print(json.dumps(behaviour, indent=2), flush=True)
    print(f'Готово: {out / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
