"""Small streaming reader shared by the three commands."""
import json
import lzma
from pathlib import Path


def parts(path):
    path = Path(path)
    if path.is_file():
        return [path]
    found = {}
    for p in sorted(path.glob('*.jsonl*')):
        if p.name.endswith(('.jsonl', '.jsonl.xz')) and not p.name.startswith('._'):
            key = p.name.removesuffix('.xz')
            if key not in found or p.suffix == '.xz':
                found[key] = p
    return [found[k] for k in sorted(found)]


def records(path):
    for part in parts(path):
        opener = lzma.open if part.suffix == '.xz' else open
        with opener(part, 'rt', encoding='utf-8') as f:
            for lineno, line in enumerate(f, 1):
                obj = json.loads(line)
                if 'row' in obj:
                    yield obj
                else:
                    yield {'file': part.name, 'line': lineno, 'row': obj}
