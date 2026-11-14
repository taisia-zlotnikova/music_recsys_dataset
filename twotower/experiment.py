"""Три режима на pilot: замер скорости, общий бюджет 2/4 эпохи, resume, полный val."""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch

from .features import file_hash

MODES = ('both', 'text', 'history')


def checkpoint_step(path):
    if not path.exists():
        return 0
    return torch.load(path, map_location='cpu', weights_only=False, mmap=True)['step']


def choose_epochs(rates, current, steps_per_epoch, seconds_left):
    # Запас 40% на колебания скорости/validation и час на итоговую оценку.
    for epochs in (4, 2):
        estimate = sum(max(0, epochs * steps_per_epoch - current[m]) * 128 / rates[m]
                       for m in MODES)
        if 1.4 * estimate + 3600 <= seconds_left:
            return epochs
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/model/pilot')
    p.add_argument('--features', default='data/model/features.npz')
    p.add_argument('--out', required=True, help='Та же папка при продолжении после разрыва Colab')
    p.add_argument('--hours', type=float, default=22, help='Бюджет текущего запуска, включая оценку')
    p.add_argument('--batch-size', type=int, choices=[8, 16, 32], default=8)
    a = p.parse_args()
    if a.hours <= 0 or not torch.cuda.is_available():
        p.error('Нужны CUDA и положительный --hours')
    started = time.monotonic()
    deadline = started + a.hours * 3600
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(a.data) / 'manifest.json').read_text())
    settings = dict(data_sha256=file_hash(Path(a.data) / 'manifest.json'),
                    feature_sha256=file_hash(a.features), batch_size=a.batch_size)
    settings_file = out / 'experiment.json'
    if settings_file.exists():
        saved = json.loads(settings_file.read_text())
        if saved != settings:
            raise ValueError('Папка эксперимента уже использует другие данные/батч; не меняй их при resume')
    else:
        settings_file.write_text(json.dumps(settings, indent=2) + '\n')
    steps_per_epoch = math.ceil(math.ceil(manifest['splits']['train']['examples'] / a.batch_size)
                                / (128 // a.batch_size))
    common = [sys.executable, '-u', '-m', 'twotower.train', '--device', 'cuda',
              '--data', a.data, '--features', a.features, '--tuning', 'adapters', '--lora-r', '8',
              '--encoder-layers', '2', '--cross-every', '4', '--item-features', 'audio,text,tags',
              '--batch-size', str(a.batch_size), '--accumulation', str(128 // a.batch_size),
              '--lr', '0.001', '--lora-lr', '0.0001', '--seed', '42', '--val-turns', '5000',
              '--eval-every', str(steps_per_epoch), '--save-every', '100']

    def execute(command):
        subprocess.run(command, check=True, timeout=max(1, deadline - time.monotonic()))

    def train(mode, target):
        folder = out / mode
        last = folder / 'last.pt'
        if checkpoint_step(last) >= target:
            return True
        remaining = deadline - time.monotonic() - 600
        if remaining <= 0:
            return False
        command = common + ['--mode', mode, '--out', str(folder), '--steps', str(target),
                            '--max-seconds', str(remaining)]
        if last.exists():
            command += ['--resume', str(last)]
        elif folder.exists():
            # train.py создаёт каталог сам; не удаляем результаты неудачного старта.
            if any(folder.iterdir()):
                raise RuntimeError(f'{folder}: есть файлы, но нет last.pt. Выбери новую папку --out.')
            folder.rmdir()
        print(f'\n===== {mode}: до шага {target} =====', flush=True)
        execute(command)
        return checkpoint_step(last) >= target

    plan_file = out / 'plan.json'
    if not plan_file.exists():
        for mode in MODES:
            if not train(mode, 20):
                print('Бюджет исчерпан на замере. Результаты сохранены.', flush=True)
                return
        rates, current = {}, {}
        for mode in MODES:
            logs = [json.loads(line) for line in (out / mode / 'metrics.jsonl').read_text().splitlines()]
            rates[mode] = logs[-1]['examples_per_second']
            current[mode] = checkpoint_step(out / mode / 'last.pt')
        epochs = choose_epochs(rates, current, steps_per_epoch, deadline - time.monotonic())
        print(f'Замер, ходов/сек: {rates}', flush=True)
        if epochs is None:
            print('Даже 2 эпохи трёх режимов не помещаются по замеру. Длинное обучение НЕ запущено. '
                  'Checkpoint первых 20 шагов сохранены; нужен меньший набор или больший бюджет.', flush=True)
            return
        plan = dict(epochs=epochs, steps_per_epoch=steps_per_epoch, measured_rates=rates,
                    gpu=torch.cuda.get_device_name(), hours=a.hours)
        plan_file.write_text(json.dumps(plan, indent=2) + '\n')
    else:
        plan = json.loads(plan_file.read_text())
    print(f'План: {plan["epochs"]} эпохи × 3 режима, {steps_per_epoch} шагов/эпоху.', flush=True)
    # По эпохе на каждый режим: при обрыве никто не остаётся совсем без обучения.
    for epoch in range(1, plan['epochs'] + 1):
        for mode in MODES:
            if not train(mode, epoch * steps_per_epoch):
                print('Остановлено по времени. Повтори ту же команду для продолжения; '
                      'смотри last.pt и metrics.jsonl. Эксперимент ещё не завершён.', flush=True)
                return
    # Все режимы оцениваем на одинаковых 8421 ходах; итоговые файлы не путаем с val=5000.
    for mode in MODES:
        result = out / mode / 'evaluation.json'
        if not result.exists():
            execute(common + ['--evaluate', str(out / mode / 'best.pt'), '--val-turns', '0'])
    baseline = out / 'baselines_all_val.json'
    if not baseline.exists():
        execute([sys.executable, '-u', '-m', 'twotower.baselines', '--data', a.data,
                 '--features', a.features, '--val-turns', '0', '--out', str(baseline)])
    summary = {mode: json.loads((out / mode / 'evaluation.json').read_text()) for mode in MODES}
    summary['baselines'] = json.loads(baseline.read_text())
    (out / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Готово: {out / "results.json"}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except subprocess.TimeoutExpired:
        print('Лимит времени процесса исчерпан. Эксперимент НЕ завершён. '
              'Повтори ту же команду: продолжение с последнего сохранённого checkpoint.', flush=True)
