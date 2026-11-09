#!/usr/bin/env python3
"""Basic descriptive statistics for v2; does not clean or split the dataset."""
import argparse
import csv
import heapq
import html
import json
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from dataset_io import parts, records


def keep(heap, priority, row, size):
    entry = (priority, row)
    if len(heap) < size:
        heapq.heappush(heap, entry)
    elif priority > heap[0][0]:
        heapq.heapreplace(heap, entry)


def scan(job):
    path, subset, size, seed = job
    rng = random.Random(seed)
    counts, labels, users, tracks = Counter(), Counter(), Counter(), Counter()
    hist = {k: Counter() for k in ['tracks_per_dialog', 'turns_per_dialog', 'tracks_per_turn', 'lines_per_turn', 'words_per_line']}
    starts, ends, sample = Counter(), Counter(), []
    for item in records(path):
        row = item['row']
        counts['input_rows'] += 1
        label = 'true' if row.get('clean_v2') is True else 'false' if row.get('clean_v2') is False else 'missing'
        labels[label] += 1
        if subset == 'clean' and label != 'true':
            continue
        counts['dialogs'] += 1
        users[row['user_id']] += 1
        tracks.update(row['tracks'])
        counts['track_occurrences'] += len(row['tracks'])
        hist['tracks_per_dialog'][len(row['tracks'])] += 1
        turns = row.get('turns') or []
        counts['dialogs_without_turns'] += not turns
        hist['turns_per_dialog'][len(turns)] += 1
        counts['turns'] += len(turns)
        counts['multi_turn_dialogs'] += len(turns) > 1
        for lines, k in turns:
            hist['tracks_per_turn'][k] += 1
            hist['lines_per_turn'][len(lines)] += 1
            counts['user_lines'] += len(lines)
            for text in lines:
                words = len(text.split())
                hist['words_per_line'][words] += 1
                counts['words'] += words
        starts[row['ts_start'] // 86400] += 1
        ends[row['ts_end'] // 86400] += 1
        counts['duration_seconds'] += row['ts_end'] - row['ts_start']
        keep(sample, rng.random(), item, size)
    return counts, labels, users, tracks, hist, starts, ends, sample


def describe(hist):
    n = sum(hist.values())
    if not n:
        return {'n': 0}
    acc, median = 0, None
    for value, count in sorted(hist.items()):
        acc += count
        if acc * 2 >= n:
            median = value
            break
    return {'n': n, 'mean': sum(k*v for k, v in hist.items())/n, 'median': median, 'min': min(hist), 'max': max(hist)}


def csv_file(path, header, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def split_options(starts, ends, subset):
    """Counts for discussion only; never exports or assigns dataset rows."""
    n = sum(starts.values())
    day_string = lambda d: datetime.fromtimestamp(d*86400,timezone.utc).date().isoformat()
    proposed = []
    for fraction in [.8,.9,.95]:
        acc = 0
        for day,count in sorted(starts.items()):
            acc += count
            if acc >= n*fraction:
                proposed.append((f'{round(fraction*100)}/{round((1-fraction)*100)}',day+1))
                break
    for year in [2017,2018,2019]:
        day = int(datetime(year,1,1,tzinfo=timezone.utc).timestamp())//86400
        proposed.append(('calendar',day))
    options = []
    for name,cutoff in proposed:
        train = sum(c for d,c in ends.items() if d<cutoff)
        val = sum(c for d,c in starts.items() if d>=cutoff)
        options.append({'option':name,'cutoff_utc':day_string(cutoff),'train':train,'validation':val,
                        'crossing':n-train-val,'validation_pct_retained':100*val/(train+val)})
    return {'subset':subset,'status':'discussion_only_no_split_created','options':options}


def report_text(summary, hist):
    c = summary['counts']
    lines = ['# Music by Prompt v2: базовая статистика', '',
             f"Выборка: **{summary['subset']}**. Входных строк: {c['input_rows']:,}. Диалог целиком является единицей наблюдения.", '',
             '| Показатель | Значение |', '|---|---:|']
    for key, name in [('dialogs','Диалоги'),('turns','Ходы / выдачи'),('user_lines','Реплики пользователя'),('track_occurrences','Треков в сессиях'),('multi_turn_dialogs','Диалоги с несколькими ходами')]:
        lines.append(f'| {name} | {c[key]:,} |')
    lines += [f"| Пользователи | {summary['unique_users']:,} |", f"| Уникальные треки | {summary['unique_tracks']:,} |", '',
              f"Даты начала сессий (UTC): {summary['start_date_range'][0]} — {summary['start_date_range'][1]}.", '',
              '| Распределение | Среднее | Медиана | Минимум | Максимум |', '|---|---:|---:|---:|---:|']
    for key, name in [('tracks_per_dialog','Треков / диалог'),('turns_per_dialog','Ходов / диалог'),('tracks_per_turn','Треков / ход'),('lines_per_turn','Реплик / ход'),('words_per_line','Слов / реплика')]:
        d = summary['distributions'][key]
        if d['n']:
            lines.append(f"| {name} | {d['mean']:.2f} | {d['median']} | {d['min']} | {d['max']} |")
    for key, title in [('tracks_per_dialog','Треков в диалоге'),('turns_per_dialog','Ходов в диалоге')]:
        lines += ['', '## '+title, '', '| Количество | Диалоги | Доля |', '|---|---:|---:|']
        for value, n in sorted(hist[key].items()):
            lines.append(f'| {value} | {n:,} | {100*n/c["dialogs"]:.2f}% |')
    lines += ['', 'Число треков в сессиях — количество элементов tracks, а не число всех исходных прослушиваний. Ход содержит реплики user_lines и одну выдачу из k треков.', '',
              'clean_v2 — исходная метка датасета. Дополнительные проверки и чистка не выполняются. Разбиение не создаётся.']
    return '\n'.join(lines)


def report_html(summary, hist):
    c = summary['counts']
    blocks = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Music by Prompt v2: статистика</title>',
              '<style>body{max-width:1000px;margin:30px auto;padding:16px;font:16px/1.6 system-ui;color:#182230}table{border-collapse:collapse;width:100%;margin:16px 0 32px}td,th{padding:7px 12px;text-align:left;border-bottom:1px solid #ddd}.bar{height:12px;background:#4e78c4;border-radius:3px;min-width:1px}h2{margin-top:36px}</style>',
              '<h1>Music by Prompt v2</h1>',f"<p>Выборка: <b>{html.escape(summary['subset'])}</b>. Сессии/диалоги учитываются целиком. Дополнительная чистка и разбиение не выполнялись.</p>",
              '<table><tr><th>Показатель</th><th>Количество</th></tr>']
    for label,n in [('Входных строк',c['input_rows']),('Диалоги',c['dialogs']),('Ходы / выдачи',c['turns']),('Реплики',c['user_lines']),('Треков в сессиях',c['track_occurrences']),('Пользователи',summary['unique_users']),('Уникальные треки',summary['unique_tracks'])]:
        blocks.append(f'<tr><td>{label}</td><td>{n:,}</td></tr>')
    blocks.append('</table>')
    for key,title in [('tracks_per_dialog','Треков в диалоге'),('turns_per_dialog','Ходов в диалоге')]:
        blocks.append(f'<h2>{title}</h2><table><tr><th>Количество</th><th>Диалоги</th><th>Доля</th><th style="width:40%"></th></tr>')
        maximum = max(hist[key].values())
        for value,n in sorted(hist[key].items()):
            blocks.append(f'<tr><td>{value}</td><td>{n:,}</td><td>{100*n/c["dialogs"]:.2f}%</td><td><div class="bar" style="width:{100*n/maximum:.2f}%"></div></td></tr>')
        blocks.append('</table>')
    blocks.append('<h2>Средние и медианы</h2><table><tr><th>Распределение</th><th>Среднее</th><th>Медиана</th></tr>')
    for key,name in [('tracks_per_dialog','Треки / диалог'),('turns_per_dialog','Ходы / диалог'),('tracks_per_turn','Треки / ход'),('lines_per_turn','Реплики / ход'),('words_per_line','Слова / реплика')]:
        d = summary['distributions'][key]
        if d['n']:
            blocks.append(f'<tr><td>{name}</td><td>{d["mean"]:.2f}</td><td>{d["median"]}</td></tr>')
    blocks.append('</table><p>Даты начала сессий UTC: '+' — '.join(summary['start_date_range'])+'.</p></html>')
    return ''.join(blocks)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', nargs='?', default='data/dialogs_v2')
    p.add_argument('--subset', choices=['all','clean'], default='clean', help='clean = clean_v2 is true')
    p.add_argument('--out', default='reports')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--sample-size', type=int, default=5000)
    p.add_argument('--split-options',action='store_true',help='Also count date/dimension options; does not split rows')
    args = p.parse_args()
    if args.workers < 1 or args.sample_size < 1:
        p.error('workers/sample-size must be positive')
    inputs = parts(args.input)
    if not inputs:
        p.error('No JSONL parts found')
    counts, labels, users, tracks = Counter(), Counter(), Counter(), Counter()
    hist = {k: Counter() for k in ['tracks_per_dialog','turns_per_dialog','tracks_per_turn','lines_per_turn','words_per_line']}
    starts, ends, sample = Counter(), Counter(), []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for path, result in zip(inputs, pool.map(scan, [(p,args.subset,args.sample_size,42+i) for i,p in enumerate(inputs)])):
            c,l,u,t,h,s,e,items = result
            for dest, src in [(counts,c),(labels,l),(users,u),(tracks,t),(starts,s),(ends,e)]:
                dest.update(src)
            for key in hist:
                hist[key].update(h[key])
            for priority, item in items:
                keep(sample, priority, item, args.sample_size)
            print(f'{path.name}: {c["dialogs"]:,} selected dialogs', flush=True)
    if not counts['dialogs']:
        p.error('No dialogs selected')
    date = lambda day: datetime.fromtimestamp(day*86400,timezone.utc).date().isoformat()
    summary = {'input':str(Path(args.input).resolve()),'subset':'clean_v2=true' if args.subset=='clean' else 'all',
               'counts':dict(counts),'input_clean_v2':dict(labels),'unique_users':len(users),'unique_tracks':len(tracks),
               'distributions':{k:describe(v) for k,v in hist.items()},'start_date_range':[date(min(starts)),date(max(starts))],
               'sample_size':len(sample),'sample_seed':42}
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    text = report_text(summary,hist)
    (out/'report.md').write_text(text,encoding='utf-8')
    (out/'report.html').write_text(report_html(summary,hist),encoding='utf-8')
    for key, values in hist.items():
        csv_file(out/(key+'.csv'),['value','count'],sorted(values.items()))
    csv_file(out/'tracks.csv',['track_id','events'],tracks.most_common())
    csv_file(out/'timeline.csv',['date_utc','starts','ends'],[(date(d),starts[d],ends[d]) for d in sorted(starts.keys()|ends.keys())])
    with (out/'sample.jsonl').open('w',encoding='utf-8') as f:
        for _, item in sorted(sample,reverse=True):
            f.write(json.dumps(item,ensure_ascii=False)+'\n')
    if args.split_options:
        (out/'split_options.json').write_text(json.dumps(split_options(starts,ends,summary['subset']),ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Saved {out}/report.md and summary.json')


if __name__ == '__main__':
    main()
