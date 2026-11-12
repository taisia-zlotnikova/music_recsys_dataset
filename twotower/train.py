"""Обучение на подготовленных ходах: full softmax, validation, checkpoint/resume."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import Collator, ShardSampler, TurnDataset
from .features import check_catalog, file_hash
from .model import DEFAULT_ITEM_FEATURES, build_model, load_trainable, positive_loss, retrieval_metrics, trainable_state


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def autocast(device):
    return torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda')


def select_device(name='auto'):
    # MPS выбирается явно: основной проверяемый GPU-путь — CUDA.
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA недоступна. В Colab выбери GPU в настройках среды.')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS недоступен этому процессу. Для проверки используй --device cpu.')
    return torch.device(name)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    parts, totals, count = model.item_parts(), {}, 0
    for batch in loader:
        batch = to_device(batch, device)
        with autocast(device):
            scores = model.scores(batch, parts)
        if not torch.isfinite(scores).all():
            raise FloatingPointError('Нечисловые оценки на validation')
        n = len(batch['tok'])
        values = retrieval_metrics(scores, batch['targets'])
        values['loss'] = positive_loss(scores, batch['targets']).item() * n
        for key, value in values.items():
            totals[key] = totals.get(key, 0) + value
        count += n
    model.train()
    return {key: value / count for key, value in totals.items()}


def save_checkpoint(path, model, optimizer, scaler, config, step, epoch, next_batch, best, data_hash):
    payload = {'weights': trainable_state(model), 'config': config, 'step': step,
               'epoch': epoch, 'next_batch': next_batch, 'best': best, 'data_sha256': data_hash,
               'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
               'torch_rng': torch.get_rng_state(), 'numpy_rng': np.random.get_state(),
               'python_rng': random.getstate(),
               'mps_rng': torch.mps.get_rng_state() if next(model.parameters()).device.type == 'mps' else None,
               'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    path = Path(path)
    temporary = path.with_suffix('.partial')
    torch.save(payload, temporary)
    temporary.replace(path)


def run_training(model, train_data, val_data, manifest, config, out, device,
                 steps=1000, batch_size=8, accumulation=16, eval_every=100,
                 lr=1e-3, copied_lr=1e-4, lora_lr=2e-5, seed=42, resume=None, data_hash=''):
    """steps — итоговое число optimizer steps, также при --resume."""
    out = Path(out)
    collator = Collator(manifest['pad_id'], manifest['history_only_tokens'], config['mode'])
    sampler = ShardSampler(train_data, seed)
    loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler, collate_fn=collator,
                        generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collator,
                            generator=torch.Generator().manual_seed(seed))
    groups = {'new': [], 'copied': [], 'lora': []}
    for name, p in model.named_parameters():
        if p.requires_grad:
            group = 'lora' if 'lora_' in name else 'copied' if name.startswith(('user.encoder.', 'user.cross.')) else 'new'
            groups[group].append(p)
    optimizer = torch.optim.AdamW([{'params': groups[k], 'lr': rate}
                                  for k, rate in [('new', lr), ('copied', copied_lr), ('lora', lora_lr)]
                                  if groups[k]], weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    step, epoch, skip, best = 0, 0, 0, -1.0
    if resume:
        optimizer.load_state_dict(resume['optimizer'])
        scaler.load_state_dict(resume['scaler'])
        step, epoch, skip, best = (resume[k] for k in ('step', 'epoch', 'next_batch', 'best'))
        torch.set_rng_state(resume['torch_rng'])
        np.random.set_state(resume['numpy_rng'])
        random.setstate(resume['python_rng'])
        if device.type == 'cuda' and resume['cuda_rng']:
            torch.cuda.set_rng_state_all(resume['cuda_rng'])
        if device.type == 'mps' and resume.get('mps_rng') is not None:
            torch.mps.set_rng_state(resume['mps_rng'])
    optimizer.zero_grad(set_to_none=True)
    model.train()
    started, window, window_n, processed = time.monotonic(), 0.0, 0, 0
    last_validation = None
    while step < steps:
        sampler.epoch = epoch
        for index, batch in enumerate(loader):
            if index < skip:
                continue
            # Последняя неполная группа накопления тоже даёт корректное среднее по ходам.
            group_start = (index // accumulation) * accumulation
            group_stop = min(group_start + accumulation, len(loader))
            group_examples = min(len(train_data), group_stop * batch_size) - group_start * batch_size
            batch = to_device(batch, device)
            with autocast(device):
                loss = model(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Нечисловой loss на шаге {step}')
            n = len(batch['tok'])
            scaler.scale(loss * n / group_examples).backward()
            window += loss.detach().item() * n
            window_n += n
            processed += n
            if index + 1 != group_stop:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scaler.get_scale() < old_scale:
                print('FP16 overflow: обновление пропущено, scale уменьшен', flush=True)
                continue
            step += 1
            log = {'step': step, 'train_loss': window / window_n,
                   'examples_per_second': processed / max(time.monotonic() - started, 1e-6),
                   'temperature': model.log_temperature.exp().item()}
            if device.type == 'cuda':
                log['peak_gpu_gb'] = torch.cuda.max_memory_allocated() / 1e9
            if step % eval_every == 0 or step == steps:
                metrics = evaluate(model, val_loader, device)
                last_validation = metrics
                log['val'] = metrics
                improved = metrics['ndcg@100'] > best
                best = max(best, metrics['ndcg@100'])
                for name in (['last.pt', 'best.pt'] if improved else ['last.pt']):
                    save_checkpoint(out / name, model, optimizer, scaler, config, step,
                                    epoch, index + 1, best, data_hash)
            with (out / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(log) + '\n')
            print(json.dumps(log), flush=True)
            window, window_n = 0.0, 0
            if step >= steps:
                break
        epoch += 1
        skip = 0
    return last_validation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', default='data/model/features.npz')
    p.add_argument('--item-features', default=','.join(DEFAULT_ITEM_FEATURES),
                   help='Каналы трека через запятую: audio,text,tags; можно добавлять dense-каналы из NPZ')
    p.add_argument('--data', default='data/model/pilot')
    p.add_argument('--out', default='runs/encdec')
    p.add_argument('--mode', choices=['both', 'text', 'history'], default='both')
    p.add_argument('--encoder-layers', type=int, default=2)
    p.add_argument('--cross-every', type=int, default=4)
    p.add_argument('--dim', type=int, default=256)
    p.add_argument('--lora-r', type=int, default=0)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--accumulation', type=int, default=16)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--val-turns', type=int, default=5000, help='0 = все ходы подготовленной val')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--copied-lr', type=float, default=1e-4)
    p.add_argument('--lora-lr', type=float, default=2e-5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--resume', help='Продолжить с optimizer/RNG и позицией в данных')
    p.add_argument('--init', help='Взять обучаемые веса, начать новый этап/optimizer (например LoRA)')
    p.add_argument('--evaluate', help='Только оценить checkpoint, без обучения')
    a = p.parse_args()
    if min(a.steps, a.batch_size, a.accumulation, a.eval_every) < 1 or a.val_turns < 0:
        p.error('Число шагов/батч/накопление/интервал должны быть положительными')
    if sum(bool(x) for x in (a.resume, a.init, a.evaluate)) > 1:
        p.error('Выбери только один из --resume, --init, --evaluate')
    device = select_device(a.device)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    root = Path(a.data)
    manifest = json.loads((root / 'manifest.json').read_text())
    feature_hash, data_hash = file_hash(a.features), file_hash(root / 'manifest.json')
    check_catalog(root, a.features, manifest)
    config = {k: getattr(a, k) for k in ['mode', 'dim', 'encoder_layers', 'cross_every', 'lora_r']}
    config.update(model=manifest['model'], history_len=manifest['history_len'],
                  item_features=[s.strip() for s in a.item_features.split(',')],
                  feature_sha256=feature_hash, batch_size=a.batch_size, accumulation=a.accumulation,
                  seed=a.seed, val_turns=a.val_turns)
    checkpoint_path = a.resume or a.init or a.evaluate
    checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False) if checkpoint_path else None
    if a.evaluate:
        config = checkpoint_data['config']
        if config['feature_sha256'] != feature_hash or checkpoint_data['data_sha256'] != data_hash:
            raise ValueError('Для оценки нужны те же признаки и подготовленные данные')
    if checkpoint_data:
        old = checkpoint_data['config']
        required = ['model', 'mode', 'dim', 'history_len', 'encoder_layers', 'cross_every', 'feature_sha256']
        if a.resume:
            required += ['lora_r', 'batch_size', 'accumulation', 'seed', 'val_turns']
            if checkpoint_data['data_sha256'] != data_hash:
                raise ValueError('Для --resume нужны те же подготовленные данные')
        elif old['lora_r'] and old['lora_r'] != config['lora_r']:
            raise ValueError('Нельзя менять размер уже обученной LoRA')
        if (any(old[k] != config[k] for k in required)
                or list(old.get('item_features', DEFAULT_ITEM_FEATURES)) != list(config.get('item_features', DEFAULT_ITEM_FEATURES))):
            raise ValueError('Настройки не совпадают с checkpoint')
    train_data, val_data = TurnDataset(root / 'train'), TurnDataset(root / 'val')
    if not len(train_data) or not len(val_data):
        raise ValueError('Нужны непустые train и val')
    if a.val_turns and a.val_turns < len(val_data):
        # Фиксированная равномерная выборка, одинаковая для всех режимов.
        indices = np.random.default_rng(config['seed']).choice(len(val_data), a.val_turns, replace=False)
        val_data = Subset(val_data, np.sort(indices).tolist())
    if a.evaluate:
        with np.load(a.features) as features:
            model = build_model(features, config, device, a.local_files_only)
        load_trainable(model, checkpoint_data['weights'])
        collator = Collator(manifest['pad_id'], manifest['history_only_tokens'], config['mode'])
        metrics = evaluate(model, DataLoader(val_data, batch_size=a.batch_size, collate_fn=collator), device)
        result = {'checkpoint': a.evaluate, 'val_turns': len(val_data), 'metrics': metrics}
        Path(a.evaluate).with_name('evaluation.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
        return
    out = Path(a.out)
    if a.resume:
        out.mkdir(parents=True, exist_ok=True)
    else:
        out.mkdir(parents=True, exist_ok=False)
    with np.load(a.features) as features:
        model = build_model(features, config, device, a.local_files_only)
    if checkpoint_data:
        load_trainable(model, checkpoint_data['weights'], allow_new_lora=bool(a.init))
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    print(f'device={device}; train={len(train_data):,}; val={len(val_data):,}; '
          f'trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}', flush=True)
    run_training(model, train_data, val_data, manifest, config, out, device,
                 a.steps, a.batch_size, a.accumulation, a.eval_every, a.lr, a.copied_lr,
                 a.lora_lr, a.seed, checkpoint_data if a.resume else None, data_hash)


if __name__ == '__main__':
    main()
