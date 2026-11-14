import numpy as np

from twotower.diagnose import aggregates, bootstrap_difference, other_users, per_turn_metrics


def test_other_users_is_reproducible_permutation():
    users = np.array([1, 1, 1, 2, 2, 3, 4])
    donors = other_users(users, 42)
    assert sorted(donors) == list(range(len(users)))
    assert np.all(users != users[donors])
    assert np.array_equal(donors, other_users(users, 42))


def test_target_groups_keep_original_positions_and_mixed_turns():
    rows = [{'targets': np.array([3, 40]), 'history': np.array([[3, 0, 0]])},
            {'targets': np.array([70]), 'history': np.array([[5, 0, 0]])}]
    top = np.tile(np.arange(100), (2, 1))
    v = per_turn_metrics(top, rows)
    assert v['recall@10'].tolist() == [0.5, 0.0]
    assert v['recall@100'].tolist() == [1.0, 1.0]
    assert v['seen_recall@10'][0] == 1
    assert np.isnan(v['seen_recall@10'][1])
    assert v['new_recall@10'].tolist() == [0.0, 0.0]
    report = aggregates(v, [{'turn': 0}, {'turn': 1}])
    assert report['all']['seen_recall@10']['turns'] == 1
    assert report['all']['new_recall@10']['turns'] == 2
    assert report['first']['recall@10']['value'] == 0.5


def test_paired_bootstrap_preserves_constant_difference():
    result = bootstrap_difference(np.array([0.4, 0.8, 0.6]), np.array([0.2, 0.6, 0.4]),
                                  ['a', 'a', 'b'], repeats=100)
    assert np.allclose(result['ci95'], [0.2, 0.2])
    assert result['dialogues'] == 2


def test_exact_ranks_full_catalog_and_padding():
    import torch
    from twotower.history_sweep import target_ranks
    scores = torch.tensor([[0., 4., 2., 4.], [3., 2., 1., 0.]])
    targets = torch.tensor([[3, 0], [0, -1]])
    assert target_ranks(scores, targets) == [[2, 4], [1]]
