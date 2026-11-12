"""Собрать готовые MuQ/Qwen и теги в общем порядке Music4All ID."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check_catalog(root, features_path, manifest):
    """Ходы зависят от ID и их порядка, а не от выбранного эмбеддера."""
    catalog = Path(root) / 'track_ids.npy'
    if catalog.exists():
        with np.load(features_path) as f:
            if not np.array_equal(np.load(catalog), f['track_ids']):
                raise ValueError('Каталог или порядок track_ids отличается от подготовленных ходов')
    elif file_hash(features_path) != manifest['feature_sha256']:
        raise ValueError('Для старых данных без track_ids.npy нужен исходный файл признаков')


def add_features(base, additions, out):
    """Подключить готовые dense-векторы по ID; не запускает текстовую/аудиомодель."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    with np.load(base) as f:
        pack = {key: f[key] for key in f.files}
    sources = {}
    reserved = {'tags', 'track_ids', 'artists', 'titles', 'tag_ids', 'tag_weights', 'tag_vocab'}
    for spec in additions:
        name, separator, path = spec.partition('=')
        if not separator or not re.fullmatch('[a-z][a-z0-9_]*', name) or name in reserved:
            raise ValueError('Ожидается --add имя=файл.npz; имя — audio, text, lyrics, tags_text и т.п.')
        with np.load(path) as f:
            ids, vectors = f['track_ids'], f['vectors']
        if vectors.ndim != 2 or len(vectors) != len(ids) or len(set(ids)) != len(ids):
            raise ValueError(f'{path}: нужны уникальные track_ids и vectors [N, D]')
        pos = {t: i for i, t in enumerate(ids)}
        missing = [t for t in pack['track_ids'] if t not in pos]
        if missing:
            raise ValueError(f'{path}: нет векторов для {len(missing)} треков')
        values = vectors[[pos[t] for t in pack['track_ids']]].astype(np.float32)
        if not np.isfinite(values).all() or values.shape[1] < 1:
            raise ValueError(f'{path}: некорректные векторы')
        # Нули разрешены: например, отсутствие lyrics у конкретного трека.
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        pack[name] = values
        sources[name] = {'path': path, 'sha256': file_hash(path), 'dim': values.shape[1]}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('wb') as f:
        np.savez(f, **pack)
    summary = {'base_sha256': file_hash(base), 'added': sources, 'sha256': file_hash(out)}
    out.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def normalize_tag(text):
    text = str(text).lower().replace('ё', 'е').strip()
    aliases = {'hiphop': 'hip hop', 'r&b': 'rnb', 'r and b': 'rnb',
               'r n b': 'rnb', 'lofi': 'lo fi', 'dnb': 'drum n bass',
               'd&b': 'drum n bass', 'drum and bass': 'drum n bass'}
    text = aliases.get(text, text)
    text = re.sub(r'[-_/]+', ' ', text)
    text = re.sub(r"[^a-z0-9а-я&' ]+", ' ', text)
    text = ' '.join(text.split())
    return aliases.get(text, text)


def tag_arrays(rows, min_count=5, max_tags=32):
    per_track = []
    for row in rows:
        weights = row.get('lastfm_tag_weights') or '{}'
        weights = json.loads(weights) if isinstance(weights, str) else weights
        tags = {}
        for tag, weight in weights.items():
            tag = normalize_tag(tag)
            if tag and float(weight) > 0:
                tags[tag] = max(tags.get(tag, 0), float(weight) / 100)
        # Если у перечисленного тега нет веса Last.fm, используем вес 1.
        for tag in (row.get('m4a_tags') or '').split(','):
            tag = normalize_tag(tag)
            if tag:
                tags.setdefault(tag, 1.0)
        for genre in set((row.get('m4a_genres') or '').split(',')):
            genre = normalize_tag(genre)
            if genre:
                tags[genre] = tags.get(genre, 0) + 0.5
        per_track.append(tags)
    counts = Counter(t for tags in per_track for t in tags)
    vocab = ['<no_tags>'] + sorted((t for t, n in counts.items() if n >= min_count),
                                   key=lambda t: (-counts[t], t))
    lookup = {t: i for i, t in enumerate(vocab)}
    indices = np.full((len(rows), max_tags), len(vocab), np.int32)
    weights = np.zeros(indices.shape, np.float32)
    for i, tags in enumerate(per_track):
        keep = sorted(((lookup[t], w) for t, w in tags.items() if t in lookup),
                      key=lambda x: (-x[1], x[0]))[:max_tags] or [(0, 1.0)]
        total = sum(w for _, w in keep)
        for j, (tag, w) in enumerate(keep):
            indices[i, j], weights[i, j] = tag, w / total
    return np.asarray(vocab), indices, weights


def build_features(tracks, recsys, out, min_count=5):
    recsys, out = Path(recsys), Path(out)
    if out.exists():
        raise FileExistsError(out)
    with open(tracks, encoding='utf-8') as f:
        ids = sorted({r['track_id'] for r in csv.DictReader(f)})
    if not ids:
        raise ValueError('Пустой каталог треков')
    audio = {r['m4a_id']: r['muq_embedding'] for r in pq.read_table(
        recsys / 'music4all_crs/track_embeddings.parquet').to_pylist()}
    meta = {r['m4a_id']: r for r in pq.read_table(
        recsys / 'music4all_crs/tracks_meta.parquet', columns=[
            'm4a_id', 'm4a_tags', 'm4a_genres', 'lastfm_tag_weights',
            'm4a_artist', 'm4a_song']).to_pylist()}
    with np.load(recsys / 'artifacts/item_text/qwen3emb.npz') as qwen:
        qpos = {t: i for i, t in enumerate(qwen['ids'])}
        missing = [t for t in ids if t not in audio or t not in meta or t not in qpos]
        if missing:
            raise ValueError(f'Нет признаков для {len(missing)} ID, например {missing[:3]}')
        text = qwen['vectors'][[qpos[t] for t in ids]].astype(np.float32)
    muq = np.asarray([audio[t] for t in ids], np.float32)
    for name, values in [('MuQ', muq), ('Qwen', text)]:
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError(f'Некорректные векторы {name}')
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if (norms <= 0).any():
            raise ValueError(f'Нулевые векторы {name}')
        values /= norms
    rows = [meta[t] for t in ids]
    vocab, tag_ids, tag_weights = tag_arrays(rows, min_count)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Без pickle: файл можно передать другой модели независимо от нашего кода.
    with out.open('wb') as f:
        np.savez(f, track_ids=np.asarray(ids), audio=muq, text=text,
                 tag_vocab=vocab, tag_ids=tag_ids, tag_weights=tag_weights,
                 artists=np.asarray([r['m4a_artist'] or '' for r in rows]),
                 titles=np.asarray([r['m4a_song'] or '' for r in rows]))
    summary = {'tracks': len(ids), 'audio_dim': muq.shape[1], 'text_dim': text.shape[1],
               'tag_vocab': len(vocab), 'max_tags': tag_ids.shape[1],
               'no_tags': int((tag_ids[:, 0] == 0).sum()), 'sha256': file_hash(out),
               'text_content': 'artist — title', 'tag_min_count': min_count}
    out.with_suffix('.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tracks', default='reports/tracks.csv')
    p.add_argument('--recsys', default='../recsys')
    p.add_argument('--out', default='data/model/features.npz')
    p.add_argument('--min-tag-count', type=int, default=5)
    p.add_argument('--base', help='Вместо сборки из recsys расширить существующий features.npz')
    p.add_argument('--add', action='append', default=[], metavar='NAME=FILE.npz',
                   help='Готовые track_ids/vectors: добавить канал или заменить audio/text; можно повторять')
    a = p.parse_args()
    if a.add and not a.base:
        p.error('--add используется вместе с --base')
    result = (add_features(a.base, a.add, a.out) if a.base else
              build_features(a.tracks, a.recsys, a.out, a.min_tag_count))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
