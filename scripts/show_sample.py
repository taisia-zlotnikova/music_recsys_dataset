#!/usr/bin/env python3
"""Readable examples with Music4All cards, in the terminal or a standalone HTML."""
import argparse
import html
import json
import random
from datetime import datetime, timezone
from pathlib import Path


def load_cards(path='reports/cards.jsonl'):
    if not Path(path).exists():
        return {}
    with open(path,encoding='utf-8') as f:
        return {r['track_id']:r for r in map(json.loads,f)}


def format_sample(sample, cards=None):
    cards = cards or {}
    row = sample.get('row',sample)
    date = lambda ts: datetime.fromtimestamp(ts,timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    lines = [f"Источник: {sample.get('file','—')}:{sample.get('line','—')}",
             f"Пользователь {row['user_id']} | clean_v2={row.get('clean_v2')} | source={row.get('source')}",
             f"{date(row['ts_start'])} → {date(row['ts_end'])}",
             f"Треков: {len(row['tracks'])}; ходов: {len(row.get('turns') or [])}"]
    offset = 0
    for n,(user_lines,k) in enumerate(row.get('turns') or [],1):
        lines += ['',f'── Ход {n} ──']
        lines += [f'Пользователь: {s}' for s in user_lines]
        lines.append(f'Выдача: {k} трек(ов)')
        for tid in row['tracks'][offset:offset+k]:
            card = cards.get(tid,{})
            title = f"{card.get('artist','?')} — {card.get('song','?')}"
            lines.append(f'  {tid} | {title}')
            if card:
                lines.append(f"    Альбом: {card.get('album_name','—')}; год: {card.get('release','—')}")
                lines.append(f"    Жанры: {card.get('genres','—')}; язык: {card.get('lang','—')}")
                lines.append(f"    Теги: {card.get('tags','—')}")
                attributes = []
                for key in ['tempo','danceability','energy','valence']:
                    value = card.get(key)
                    if value is not None:
                        try:
                            value = f'{float(value):.2f}'
                        except ValueError:
                            pass
                        attributes.append(f'{key}={value}')
                lines.append('    '+'; '.join(attributes))
        offset += k
    return '\n'.join(lines)


def show_sample(sample, cards=None):
    """Notebook helper: pass a dataset row or an envelope from sample.jsonl."""
    print(format_sample(sample,load_cards() if cards is None else cards))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample',default='reports/sample.jsonl')
    p.add_argument('--cards',default='reports/cards.jsonl')
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--min-turns',type=int,default=1)
    p.add_argument('--count',type=int,default=1)
    p.add_argument('--html',help='Save the chosen examples as HTML instead of printing')
    args = p.parse_args()
    if args.count<1 or args.min_turns<0:
        p.error('count must be positive; min-turns must be nonnegative')
    with open(args.sample,encoding='utf-8') as f:
        items = [s for s in map(json.loads,f) if len(s['row'].get('turns') or [])>=args.min_turns]
    if not items:
        p.error('No matching examples')
    chosen = random.Random(args.seed).sample(items,min(args.count,len(items)))
    cards = load_cards(args.cards)
    rendered = [format_sample(s,cards) for s in chosen]
    if args.html:
        blocks = ['<details open><summary>Пример '+str(i)+'</summary><pre>'+html.escape(text)+'</pre></details>' for i,text in enumerate(rendered,1)]
        out = Path(args.html)
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Music by Prompt v2: примеры</title><style>body{max-width:1100px;margin:30px auto;padding:16px;font:16px system-ui}pre{white-space:pre-wrap;font:14px/1.6 monospace}details{border-bottom:1px solid #ddd;margin:20px 0}</style><h1>Примеры диалогов v2 с карточками Music4All</h1>'+''.join(blocks)+'</html>',encoding='utf-8')
        print(f'{len(chosen)} examples → {out}')
    else:
        print(('\n\n'+'═'*72+'\n\n').join(rendered))


if __name__ == '__main__':
    main()
