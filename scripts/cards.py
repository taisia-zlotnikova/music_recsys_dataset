#!/usr/bin/env python3
"""Join the working track catalog with the five Music4All tables."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--music4all', default='data/music4all')
    p.add_argument('--tracks', default='reports/tracks.csv')
    p.add_argument('--out', default='reports/cards.jsonl')
    args = p.parse_args()
    with open(args.tracks,encoding='utf-8') as f:
        targets = {r['track_id'] for r in csv.DictReader(f)}
    cards, coverage = {}, {}
    for name in ['id_information.csv','id_metadata.csv','id_genres.csv','id_tags.csv','id_lang.csv']:
        path = Path(args.music4all)/name
        if not path.exists():
            coverage[name] = None
            continue
        matched = set()
        with path.open(encoding='utf-8-sig',newline='') as f:
            # Music4All .csv files are raw TSV; quotation marks are literal.
            for row in csv.DictReader(f,delimiter='\t',quoting=csv.QUOTE_NONE):
                tid = row.pop('id')
                if tid in targets:
                    matched.add(tid)
                    cards.setdefault(tid,{'track_id':tid}).update(row)
        coverage[name] = len(matched)
    out = Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8') as f:
        for tid in sorted(cards):
            f.write(json.dumps(cards[tid],ensure_ascii=False)+'\n')
    summary = {'target_tracks':len(targets),'matched_tracks':len(cards),'matched_per_table':coverage}
    out.with_name(out.stem+'_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'{len(cards):,}/{len(targets):,} tracks → {out}')


if __name__ == '__main__':
    main()
