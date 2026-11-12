"""Короткие проверки важных свойств пайплайна; без сети и больших весов."""
import json
import sqlite3

import numpy as np
import pytest
import torch
from transformers import AutoModel, Qwen3Config

from twotower.data import Collator, ShardWriter, TextEncoder, TurnDataset
from twotower.features import add_features, check_catalog, file_hash, tag_arrays
from twotower.model import (TrackTower, TwoTower, load_trainable, positive_loss, retrieval_metrics,
                           trainable_state)
from twotower.prepare import collect_events, dialogue_turns, history_array, store_dialogues, write_turns
from twotower.train import run_training


@pytest.fixture(autouse=True)
def seed():
    torch.set_num_threads(1)
    torch.manual_seed(7)


class Tokenizer:
    eos_token_id, pad_token_id = 1, 0

    def encode(self, text, **kwargs):
        return [2 + ord(c) % 90 for c in text]

    def __call__(self, texts, **kwargs):
        return {'input_ids': [self.encode(t) for t in texts]}


def features():
    rng = np.random.default_rng(7)
    return {'audio': rng.normal(size=(12, 8)).astype('float32'),
            'text': rng.normal(size=(12, 10)).astype('float32'),
            'tag_vocab': np.array(['<no_tags>', 'rock', 'pop']),
            'tag_ids': rng.integers(0, 3, (12, 2)),
            'tag_weights': np.full((12, 2), 0.5, np.float32)}


def model(encoder_layers=1, use_history=True):
    cfg = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                     num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                     head_dim=8, max_position_embeddings=512, attn_implementation='sdpa')
    return TwoTower(AutoModel.from_config(cfg), features(), dim=8, history_len=4,
                    encoder_layers=encoder_layers, cross_every=1, use_history=use_history)


def rows():
    return [{'tokens': [9, 10, 1], 'history': np.array([[2, 0, 0], [3, 1, 2]]),
             'targets': [1, 4]}, {'tokens': [11, 1], 'history': np.empty((0, 3), np.int32), 'targets': [6]}]


def batch():
    return Collator(0, [12, 1])(rows())


def test_tags_are_weighted_normalized_and_have_fallback():
    vocab, ids, weights = tag_arrays([
        {'lastfm_tag_weights': '{"Hip-Hop": 100}', 'm4a_genres': 'hip hop'}, {}], min_count=1)
    assert vocab.tolist() == ['<no_tags>', 'hip hop']
    assert ids[:, 0].tolist() == [1, 0]
    assert np.allclose(weights.sum(1), 1)


def test_feature_switches_and_new_channel():
    pack = features()
    pack['lyrics'] = np.ones((12, 6), np.float32)
    for channels in [('text',), ('tags',), ('audio', 'text'), ('audio', 'tags', 'lyrics')]:
        tower = TrackTower(pack, dim=8, channels=channels)
        reps, vectors = tower()
        assert reps.shape == vectors.shape == (12, 8)
        assert torch.allclose(vectors.norm(dim=1), torch.ones(12), atol=1e-6)
        vectors[:, 0].sum().backward()
        assert hasattr(tower, 'tags') == ('tags' in channels)
        if 'tags' in channels:
            assert tower.tags.weight.grad is not None
        for name in set(channels) - {'tags'}:
            assert getattr(tower, name + '_proj')[0].weight.grad is not None


def test_added_vectors_align_by_id_and_reuse_prepared_data(tmp_path):
    base, extra, out = (tmp_path / name for name in ('base.npz', 'extra.npz', 'extended.npz'))
    np.savez(base, track_ids=np.array(['a', 'b']), text=np.ones((2, 4), np.float32))
    np.savez(extra, track_ids=np.array(['b', 'a']), vectors=np.array([[0., 2.], [3., 0.]]))
    add_features(base, ['lyrics=' + str(extra)], out)
    with np.load(out) as f:
        assert f['track_ids'].tolist() == ['a', 'b']
        assert f['lyrics'].tolist() == [[1., 0.], [0., 1.]]
        assert f['text'].shape == (2, 4)
    manifest = {'feature_sha256': file_hash(base)}
    np.save(tmp_path / 'track_ids.npy', np.array(['a', 'b']))
    check_catalog(tmp_path, out, manifest)
    np.save(tmp_path / 'track_ids.npy', np.array(['b', 'a']))
    with pytest.raises(ValueError, match='порядок'):
        check_catalog(tmp_path, out, manifest)


def test_targets_are_not_in_current_history_or_text():
    row = {'user_id': 1, 'ts_start': 100, 'tracks': ['a', 'b', 'c'],
           'turns': [[['first'], 2], [['next'], 1]]}
    turns = list(dialogue_turns(row, 'train:1', {3: (8, 50)},
                               {'a': 0, 'b': 1, 'c': 2}, 4))
    assert turns[0]['history'] == [[3, 3, 0]]
    assert turns[0]['targets'] == [0, 1]
    assert [h[0] for h in turns[1]['history']] == [3, 0, 1]
    assert turns[1]['targets'] == [2]
    assert turns[1]['text'] == 'User: first\nUser: next'
    assert history_array({0: (2, 10)}, [0, 1], 100, 1) == [[1, 0, 0]]


def test_full_prepare_strict_cutoff_and_shards(tmp_path):
    # В Onion намеренно обратный порядок. Plays на ts_start и позже исключаются.
    onion = tmp_path / 'events.tsv'
    onion.write_text('user_id\ttrack_id\ttimestamp\n'
                     '1\tc\t2018-08-07 00:00:00\n'
                     '1\tb\t2018-08-06 00:00:00\n'
                     '1\ta\t2018-08-05 00:00:00\n'
                     '1\ta\t2018-08-04 00:00:00\n')
    paths = {}
    for split, ts, target in [('train', 1533427200, 'b'), ('val', 1533513600, 'c')]:
        path = tmp_path / f'{split}.jsonl'
        path.write_text(json.dumps({'user_id': 1, 'ts_start': ts, 'clean_v2': True,
                                    'tracks': [target], 'turns': [[['hello'], 1]]}) + '\n')
        paths[split] = path
    db = sqlite3.connect(':memory:')
    store_dialogues(db, paths, {'train': 1, 'val': 1}, 42, 0)
    stats = collect_events(db, onion, {'a': 0, 'b': 1, 'c': 2})
    encoder = TextEncoder(Tokenizer(), 128)
    write_turns(db, {'a': 0, 'b': 1, 'c': 2}, encoder, tmp_path / 'ready', 4, 1)
    train, val = TurnDataset(tmp_path / 'ready/train'), TurnDataset(tmp_path / 'ready/val')
    # Train cutoff 2018-08-05: a на границе не виден; более ранний a виден.
    assert train[0]['history'].tolist() == [[0, 0, 1]]
    # Val cutoff 2018-08-06: оба a уже доступны, b на самой границе ещё нет.
    assert val[0]['history'].tolist() == [[0, 1, 1]]
    assert val[0]['targets'].tolist() == [2]
    assert stats['raw_events_scanned'] == 4
    assert stats['events_kept'] == 2


def test_collator_modes_and_eos():
    b = batch()
    assert b['tok'][:, -1].tolist() == [1, 1]
    assert b['hmask'].tolist() == [[True, True], [False, False]]
    assert b['targets'].tolist() == [[1, 4], [6, -1]]
    assert Collator(0, [12, 1], 'text')(rows())['hist'].shape == (2, 0)
    assert Collator(0, [12, 1], 'history')(rows())['tok'].tolist() == [[12, 1], [12, 1]]
    encoder = TextEncoder(Tokenizer(), 128)
    tok = encoder.encode(['a' * 300 + 'XYZ'])[0]
    assert len(tok) == 128 and tok[-1] == 1
    assert tok[-4:-1] == Tokenizer().encode('XYZ')


def test_zero_gates_match_huggingface_qwen_and_ignore_padding():
    m, b = model().eval(), batch()
    with torch.no_grad():
        expected = m.user.base(input_ids=b['tok'], attention_mask=b['mask'],
                               position_ids=(b['mask'].long().cumsum(1)-1).clamp_min(0)).last_hidden_state[:, -1]
        expected = torch.nn.functional.normalize(m.user.proj(expected), dim=-1)
        actual = m.user(b, m.item_parts()[0])
        assert torch.allclose(actual, expected, atol=1e-5)
        for layer in m.user.cross.values():
            layer.gate.fill_(0.5)
        with_history = m.user(b, m.item_parts()[0])
        changed = {k: v.clone() for k, v in b.items()}
        changed['hist'][1] = 11  # Только паддинг: не должен менять результат.
        assert torch.allclose(with_history, m.user(changed, m.item_parts()[0]), atol=1e-5)
        changed['hist'][0] = 11
        assert not torch.allclose(with_history[0], m.user(changed, m.item_parts()[0])[0])


def test_multi_positive_loss_and_metrics():
    scores = torch.tensor([[4., 3., 2., 1.], [1., 2., 3., 4.]], requires_grad=True)
    targets = torch.tensor([[0, 1], [3, -1]])
    expected = -(scores.log_softmax(1)[0, :2].mean() + scores.log_softmax(1)[1, 3]) / 2
    assert torch.allclose(positive_loss(scores, targets), expected)
    values = retrieval_metrics(scores, targets, (1, 2))
    assert values['recall@1'] == 1.5 and values['hit@1'] == 2
    assert values['recall@2'] == 2 and values['ndcg@2'] == 2
    positive_loss(scores, targets).backward()
    assert torch.isfinite(scores.grad).all()


def test_backward_empty_history_and_checkpoint():
    m, b = model(), batch()
    loss = m(b)
    loss.backward()
    assert all(p.grad is None for p in m.user.base.parameters())
    assert m.item.tags.weight.grad is not None
    assert all(layer.gate.grad is not None for layer in m.user.cross.values())
    saved = trainable_state(m)
    assert not any(k.startswith('user.base.') for k in saved)
    fresh = model()
    load_trainable(fresh, saved)
    # Предобученная часть должна быть одна и та же (тут случайная, копируем для сравнения).
    fresh.user.base.load_state_dict(m.user.base.state_dict())
    with torch.no_grad():
        assert torch.allclose(m.eval().scores(b), fresh.eval().scores(b), atol=1e-5)
        assert torch.isfinite(m.scores(Collator(0, [12, 1], 'text')(rows()))).all()


def test_short_training_resume_and_lora(tmp_path):
    writer = ShardWriter(tmp_path / 'train', 2)
    for r in rows() * 3:
        writer.add(r)
    writer.finish()
    dataset = TurnDataset(tmp_path / 'train')
    manifest = {'pad_id': 0, 'history_only_tokens': [12, 1]}
    config = {'mode': 'both'}
    out = tmp_path / 'run'
    out.mkdir()
    m = model()
    metrics = run_training(m, dataset, dataset, manifest, config, out, torch.device('cpu'),
                           steps=2, batch_size=2, accumulation=2, eval_every=1)
    assert 0 <= metrics['recall@10'] <= 1
    saved = torch.load(out / 'last.pt', weights_only=False)
    assert saved['step'] == 2 and (out / 'best.pt').exists()
    fresh = model()
    fresh.user.base.load_state_dict(m.user.base.state_dict())
    load_trainable(fresh, saved['weights'])
    run_training(fresh, dataset, dataset, manifest, config, out, torch.device('cpu'),
                 steps=3, batch_size=2, accumulation=2, eval_every=1, resume=saved)
    assert torch.load(out / 'last.pt', weights_only=False)['step'] == 3
    from peft import LoraConfig, get_peft_model
    lora = model()
    lora.user.base = get_peft_model(lora.user.base, LoraConfig(r=2, lora_alpha=4,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']))
    load_trainable(lora, saved['weights'], allow_new_lora=True)
    assert any('lora_' in n for n in trainable_state(lora))
    loss = lora(batch())
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for n, p in lora.named_parameters() if 'lora_' in n)
