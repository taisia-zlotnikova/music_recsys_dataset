"""По одному изменению относительно both/64: history32, no_tags, no_audio. Четыре эпохи."""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch

from .experiment import checkpoint_step
from .features import file_hash

JOBS = {'history32': (32, 'audio,text,tags'),
        'no_tags': (64, 'audio,text'), 'no_audio': (64, 'text,tags')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jobs', nargs='+', choices=JOBS, default=['history32'])
    p.add_argument('--data', default='data/model/pilot')
    p.add_argument('--features', default='data/model/features.npz')
    p.add_argument('--out', required=True)
    p.add_argument('--batch-size', type=int, choices=[8,16,32], default=8)
    p.add_argument('--hours', type=float, default=22)
    a = p.parse_args()
    if not torch.cuda.is_available() or a.hours <= 0 or len(set(a.jobs)) != len(a.jobs):
        p.error('Нужны CUDA, положительный бюджет и неповторяющиеся jobs')
    deadline = time.monotonic() + 3600*a.hours
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(a.data)/'manifest.json').read_text())
    if manifest['history_len'] != 64:
        raise ValueError('Эти абляции рассчитаны на подготовленный pilot с историей 64')
    steps_epoch = math.ceil(math.ceil(manifest['splits']['train']['examples']/a.batch_size)/(128//a.batch_size))
    target = 4*steps_epoch
    common = [sys.executable,'-u','-m','twotower.train','--device','cuda','--mode','both',
              '--data',a.data,'--features',a.features,'--tuning','adapters','--lora-r','8',
              '--encoder-layers','2','--cross-every','4','--batch-size',str(a.batch_size),
              '--accumulation',str(128//a.batch_size),'--lr','0.001','--lora-lr','0.0001',
              '--seed','42','--val-turns','5000','--eval-every',str(steps_epoch),'--save-every','100']
    def execute(command):
        subprocess.run(command,check=True,timeout=max(1,deadline-time.monotonic()))
    def train(job, steps):
        folder=out/job; last=folder/'last.pt'
        if checkpoint_step(last)>=steps:
            return True
        remaining=deadline-time.monotonic()-600
        if remaining<=0:
            return False
        limit,channels=JOBS[job]
        command=common+['--history-limit',str(limit),'--item-features',channels,'--out',str(folder),
                        '--steps',str(steps),'--max-seconds',str(remaining)]
        if last.exists():
            command+=['--resume',str(last)]
        elif folder.exists():
            if any(folder.iterdir()):
                raise RuntimeError(f'{folder}: есть файлы, но нет last.pt. Выбери новую --out для нового запуска.')
            folder.rmdir()
        print(f'\n===== {job}: до шага {steps}/{target} =====',flush=True)
        execute(command)
        return checkpoint_step(last)>=steps
    # Отдельная спецификация на каждый job: можно позже запустить другие job в той же общей папке.
    for job in a.jobs:
        settings={'data_sha256':file_hash(Path(a.data)/'manifest.json'),'feature_sha256':file_hash(a.features),
                  'batch_size':a.batch_size,'history_limit':JOBS[job][0],'item_features':JOBS[job][1],
                  'epochs':4,'steps':target,'seed':42}
        spec=out/f'{job}_plan.json'
        if spec.exists() and json.loads(spec.read_text())!=settings:
            raise ValueError(f'{job}: другая конфигурация. Для resume сохраняй прежний batch/данные.')
        spec.write_text(json.dumps(settings,indent=2)+'\n')
        if not train(job,20):
            print('Бюджет исчерпан. Замер сохранён; обучение не завершено.',flush=True)
            return
    estimated=0
    for job in a.jobs:
        remaining=max(0,target-checkpoint_step(out/job/'last.pt'))
        logs=[json.loads(l) for l in (out/job/'metrics.jsonl').read_text().splitlines()]
        estimated+=remaining*128/logs[-1]['examples_per_second']
    print(f'Оценка оставшегося чистого обучения: {estimated/3600:.2f} ч; добавляем 40% запаса и час на итоговую оценку.',flush=True)
    if 1.4*estimated+3600>deadline-time.monotonic():
        print('Четыре эпохи выбранных запусков не помещаются по замеру. Длинное обучение не запущено. '
              'Запусти один job или увеличь бюджет; первые шаги сохранены.',flush=True)
        return
    for epoch in range(1,5):
        for job in a.jobs:
            if not train(job,epoch*steps_epoch):
                print('Остановлено по времени. Повтори команду для resume. Результат ещё не готов.',flush=True)
                return
    for job in a.jobs:
        folder=out/job
        if not (folder/'evaluation.json').exists():
            execute(common+['--evaluate',str(folder/'best.pt'),'--val-turns','0'])
    results={job:json.loads((out/job/'evaluation.json').read_text()) for job in a.jobs}
    (out/('results_'+'_'.join(a.jobs)+'.json')).write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    try:
        main()
    except subprocess.TimeoutExpired:
        print('Лимит текущего запуска исчерпан. Повтори команду с той же папкой для resume.',flush=True)
