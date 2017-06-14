#!/usr/bin/env python

import h5py
import numpy as np
import os
import sys

import ABXpy.task
import ABXpy.misc.items as items


# a cleanup decorator for test methods
class cleanup(object):
    def __init__(self, f):
        self.f = f

    def __call__(self):
        try:
            self.f()
        finally:
            pass
            # try:
            #     os.remove('data.abx')
            # except:
            #     pass


# not optimise, but unimportant
def tables_equivalent(t1, t2):
    assert t1.shape == t2.shape
    for a1 in t1:
        res = False
        for a2 in t2:
            if np.array_equal(a1, a2):
                res = True
        if not res:
            return False
    return True


def get_triplets(hdf5file, by):
    triplet_db = hdf5file['triplets']
    triplets = triplet_db['data']
    by_index = list(hdf5file['bys']).index(by)
    triplets_index = triplet_db['by_index'][by_index]
    return triplets[slice(*triplets_index)]


def get_pairs(hdf5file, by):
    pairs_db = hdf5file['unique_pairs']
    pairs = pairs_db['data']
    pairs_index = pairs_db.attrs[by][1:3]
    return pairs[slice(*pairs_index)]


@cleanup
def test_basic():
    task = ABXpy.task.Task('data.item', 'c0', 'c1', 'c2')
    assert task.stats['nb_blocks'] == 8
    assert task.stats['nb_triplets'] == 8
    assert task.stats['nb_by_levels'] == 2

    task.generate_triplets()

    with h5py.File('data.abx', 'r') as f:
        triplets = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0], [3, 2, 1]])
        assert tables_equivalent(triplets, get_triplets(f, '0'))
        assert tables_equivalent(triplets, get_triplets(f, '1'))

        pairs = {2, 6, 7, 3, 8, 12, 13, 9}
        assert pairs == set(get_pairs(f, '0')[:, 0])
        assert pairs == set(get_pairs(f, '1')[:, 0])


@cleanup
def test_sample():
    task = ABXpy.task.Task('data.item', 'c0', 'c1', 'c2')
    assert task.stats['nb_blocks'] == 8
    assert task.stats['nb_triplets'] == 8
    assert task.stats['nb_by_levels'] == 2

    task.generate_triplets(sample=8)

    with h5py.File('data.abx', 'r') as f:
        triplets = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0], [3, 2, 1]])
        assert tables_equivalent(triplets, get_triplets(f, '0'))
        assert tables_equivalent(triplets, get_triplets(f, '1'))

        pairs = {2, 6, 7, 3, 8, 12, 13, 9}
        assert pairs == set(get_pairs(f, '0')[:, 0])
        assert pairs == set(get_pairs(f, '1')[:, 0])


def main():
    items.generate_testitems(2, 3, name='data.item')
    try:
        # test_basic()
        test_sample()
    finally:
        os.remove('data.item')


if __name__ == '__main__':
    main()
