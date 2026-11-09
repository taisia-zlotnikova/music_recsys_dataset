#!/usr/bin/env python3
"""Split whole clean_v2 dialogs by time into two gzip-compressed JSONL files."""
import argparse
import gzip
import json
import lzma
from datetime import datetime, timezone
from pathlib import Path

from dataset_io import parts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', nargs='?', default='data/dialogs_v2')
    p.add_argument('--cutoff', default='2018-08-06', help='YYYY-MM-DD, midnight UTC')
    p.add_argument('--out', default='data/split_v2')
    args = p.parse_args()
    cutoff = int(datetime.strptime(args.cutoff, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())
    inputs = parts(args.input)
    if not inputs:
        p.error('No JSONL parts found')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_path, val_path = out/'train.jsonl.gz', out/'val.jsonl.gz'
    train_tmp, val_tmp = out/'train.jsonl.gz.partial', out/'val.jsonl.gz.partial'
    if any(path.exists() for path in [train_path,val_path,train_tmp,val_tmp]):
        p.error('Output files already exist; choose a new --out directory')
    counts = {'train':0,'val':0,'crossing':0,'not_clean_v2':0}
    with gzip.open(train_tmp,'wb',compresslevel=1) as train, gzip.open(val_tmp,'wb',compresslevel=1) as val:
        for path in inputs:
            opener = lzma.open if path.suffix=='.xz' else open
            with opener(path,'rb') as f:
                for line in f:
                    if not line.endswith(b'\n'):
                        line += b'\n'
                    row = json.loads(line)
                    if row.get('clean_v2') is not True:
                        counts['not_clean_v2'] += 1
                    elif row['ts_end'] < cutoff:
                        train.write(line)
                        counts['train'] += 1
                    elif row['ts_start'] >= cutoff:
                        val.write(line)
                        counts['val'] += 1
                    else:
                        counts['crossing'] += 1
            print(f'{path.name}: train={counts["train"]:,}, val={counts["val"]:,}',flush=True)
    train_tmp.rename(train_path)
    val_tmp.rename(val_path)
    print(json.dumps({'cutoff_utc':args.cutoff+'T00:00:00Z',**counts},indent=2))
    print(train_path)
    print(val_path)


if __name__=='__main__':
    main()
