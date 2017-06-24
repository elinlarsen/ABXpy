"""Tests of task.py using sampling"""


import os
import pytest

import ABXpy.task
import ABXpy.distances.distances as distances
import ABXpy.score as score
import ABXpy.analyze as analyze
import ABXpy.misc.items


@pytest.fixture
def files():
    """Return a dict of ABX 'test.*' filenames

    The ABX input files test.item and test.features are randomly
    generated.

    When called, this fixtures send the dictionary of files and, once
    the test returns, remove them.

    """
    # dict of filenames
    f = {f: 'test.{}'.format(f) for f in
         ('item', 'features', 'distance', 'score', 'analyze', 'task')}
    f['task'] = f['task'].replace('.task', '.abx')

    # generate items and features
    ABXpy.misc.items.generate_db_and_feat(
        3, 3, 5, f['item'], 2, 1, f['features'])

    # send the files to the test
    yield f

    # remove the files
    for f in f.values():
        try:
            os.remove(f)
        except:
            pass


def fake_distance(x, y, normalized):
    """Distance does not matter for sampling tests"""
    return 1


def abx_pipeline(files, on, across, by, K):
    # generate the task
    task = ABXpy.task.Task(
        files['item'], on, across=across, by=by, verbose=False)
    task.generate_triplets(files['task'], max_samples=K)

    # compute fake distances
    distances.compute_distances(
        files['features'], '/features/', files['task'], files['distance'],
        fake_distance, normalized=False, n_cpu=1)

    # compute the final results
    score.score(files['task'], files['distance'], files['score'])
    analyze.analyze(files['task'], files['score'], files['analyze'])

    # from the csv analysis file, return the last column as a list of
    # integers (number of triplets per cell)
    return [int(line.split('\t')[-1].strip())
            for line in open(files['analyze'], 'r').readlines()[1:]]


# on across by parameters for the task (one test per tuple)
params = [
    # ('c0', None, None),
    ('c0', 'c1', None),
    # ('c0', 'c1', 'c2'),
    # ('c0', ['c1', 'c2'], None),
    # ('c0', None, ['c1', 'c2']),
]


@pytest.mark.parametrize('on, across, by', params)
def test_sampled(files, on, across, by):
    full = abx_pipeline(files, on, across, by, None)
    for f in ('task', 'distance', 'score'):
        os.remove(files[f])

    for K in (10000, ):
        sampled = abx_pipeline(files, on, across, by, K)

        error =  'K={}:\n'.format(K) + '\n'.join(
            '{} | {}'.format(s1, s2) for s1, s2 in zip(full, sampled))

        # same number of on-across-by cells before/after sampling
        assert len(full) == len(sampled)

        # triplets are capped in all cells
        assert all(s <= K for s in sampled), error

        # triplets are not over-capped
        assert all(s2 == K for s1, s2 in zip(full, sampled) if s1 > K), error

        # unfrequent triplets are not sampled
        assert all(s1 == s2 for s1, s2 in zip(full, sampled) if s1 <= K), error
