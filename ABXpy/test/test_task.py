"""Tests for the basic parameters of task.py"""

import collections
import h5py
import numpy as np
import pytest

# package_path = os.path.dirname(os.path.dirname(
#     os.path.dirname(os.path.realpath(__file__))))
# if not(package_path in sys.path):
#     sys.path.append(package_path)

import ABXpy.task
import ABXpy.misc.items as items

error_pairs = "pairs incorrectly generated"
error_triplets = "triplets incorrectly generated"


@pytest.fixture
def item_file(tmpdir):
    # data.item file in the test tmpdir
    name = str(tmpdir.join('data.item'))

    # put some data in the file and return its path
    items.generate_testitems(2, 3, name=name)
    return name


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


# test1, triplets and pairs verification
def test_basic(item_file):
    task = ABXpy.task.Task(item_file, 'c0', 'c1', 'c2')

    stats = task.stats
    assert stats['nb_blocks'] == 8, "incorrect stats: number of blocks"
    assert stats['nb_triplets'] == 8
    assert stats['nb_by_levels'] == 2

    task.generate_triplets()

    f = h5py.File(item_file.replace('.item', '.abx'), 'r')
    triplets_block0 = get_triplets(f, '0')
    triplets_block1 = get_triplets(f, '1')
    triplets = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0], [3, 2, 1]])
    assert tables_equivalent(triplets, triplets_block0), error_triplets
    assert tables_equivalent(triplets, triplets_block1), error_triplets

    pairs = [2, 6, 7, 3, 8, 12, 13, 9]
    pairs_block0 = get_pairs(f, '0')
    pairs_block1 = get_pairs(f, '1')
    assert (set(pairs) == set(pairs_block0[:, 0])), error_pairs
    assert (set(pairs) == set(pairs_block1[:, 0])), error_pairs


# testing with a list of across attributes, triplets verification
def test_multiple_across(item_file):
    task = ABXpy.task.Task(item_file, 'c0', ['c1', 'c2'])
    stats = task.stats
    assert stats['nb_blocks'] == 8
    assert stats['nb_triplets'] == 8
    assert stats['nb_by_levels'] == 1

    task.generate_triplets()

    f = h5py.File(item_file.replace('.item', '.abx'), 'r')
    triplets_block = get_triplets(f, '0')
    triplets = np.array([[0, 1, 6], [1, 0, 7], [2, 3, 4], [3, 2, 5],
                         [4, 5, 2], [5, 4, 3], [6, 7, 0], [7, 6, 1]])
    assert tables_equivalent(triplets, triplets_block)


# testing without any across attribute
def test_no_across(item_file):
    task = ABXpy.task.Task(item_file, 'c0', None, 'c2')

    stats = task.stats
    assert stats['nb_blocks'] == 8
    assert stats['nb_triplets'] == 16
    assert stats['nb_by_levels'] == 2

    task.generate_triplets()


# testing for multiple by attributes, asserting the statistics
def test_multiple_bys(item_file):
    items.generate_testitems(3, 4, name=item_file)
    task = ABXpy.task.Task(item_file, 'c0', None, ['c1', 'c2', 'c3'])

    stats = task.stats
    assert stats['nb_blocks'] == 81
    assert stats['nb_triplets'] == 0
    assert stats['nb_by_levels'] == 27

    task.generate_triplets()


# testing for a general filter (discarding last column)
def test_filter(item_file):
    items.generate_testitems(2, 4, name=item_file)
    task = ABXpy.task.Task(
        item_file, 'c0', 'c1', 'c2', filters=["[attr == 0 for attr in c3]"])

    stats = task.stats
    assert stats['nb_blocks'] == 8, "incorrect stats: number of blocks"
    assert stats['nb_triplets'] == 8
    assert stats['nb_by_levels'] == 2

    abx_file = item_file.replace('.item', '.abx')
    task.generate_triplets(output=abx_file)
    f = h5py.File(abx_file, 'r')

    triplets_block0 = get_triplets(f, '0')
    triplets_block1 = get_triplets(f, '1')
    triplets = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0], [3, 2, 1]])
    assert tables_equivalent(triplets, triplets_block0), error_triplets
    assert tables_equivalent(triplets, triplets_block1), error_triplets

    pairs = [2, 6, 7, 3, 8, 12, 13, 9]
    pairs_block0 = get_pairs(f, '0')
    pairs_block1 = get_pairs(f, '1')
    assert (set(pairs) == set(pairs_block0[:, 0])), error_pairs
    assert (set(pairs) == set(pairs_block1[:, 0])), error_pairs


# testing with simple filter on A, verifying triplet generation
def test_filter_on_A(item_file):
    items.generate_testitems(2, 2, name=item_file)
    task = ABXpy.task.Task(
        item_file, 'c0', filters=["[attr == 0 for attr in c0_A]"])

    stats = task.stats
    assert stats['nb_blocks'] == 4, "incorrect stats: number of blocks"
    assert stats['nb_triplets'] == 4
    assert stats['nb_by_levels'] == 1

    task.generate_triplets()
    f = h5py.File(item_file.replace('.item', '.abx'), 'r')
    triplets_block0 = get_triplets(f, '0')
    triplets = np.array([[0, 1, 2], [0, 3, 2], [2, 1, 0], [2, 3, 0]])
    assert tables_equivalent(triplets, triplets_block0), error_triplets


# testing with simple filter on B, verifying triplet generation
def test_filter_on_B(item_file):
    items.generate_testitems(2, 2, name=item_file)
    task = ABXpy.task.Task(
        item_file, 'c0', filters=["[attr == 0 for attr in c1_B]"])

    stats = task.stats
    assert stats['nb_blocks'] == 4, "incorrect stats: number of blocks"
    assert stats['nb_triplets'] == 4
    assert stats['nb_by_levels'] == 1

    task.generate_triplets()
    f = h5py.File(item_file.replace('.item', '.abx'), 'r')
    triplets_block0 = get_triplets(f, '0')
    triplets = np.array([[0, 1, 2], [1, 0, 3], [2, 1, 0], [3, 0, 1]])
    assert tables_equivalent(triplets, triplets_block0), error_triplets


# testing with simple filter on B, verifying triplet generation
def test_filter_on_C(item_file):
    items.generate_testitems(2, 2, name=item_file)
    task = ABXpy.task.Task(
        item_file, 'c0', filters=["[attr == 0 for attr in c1_X]"])

    stats = task.stats
    assert stats['nb_blocks'] == 4, "incorrect stats: number of blocks"
    assert stats['nb_triplets'] == 4
    assert stats['nb_by_levels'] == 1

    task.generate_triplets()
    f = h5py.File(item_file.replace('.item', '.abx'), 'r')
    triplets_block0 = get_triplets(f, '0')
    triplets = np.array([[2, 1, 0], [2, 3, 0], [3, 0, 1], [3, 2, 1]])
    assert tables_equivalent(triplets, triplets_block0), error_triplets



def test_sample_all_triplets(item_file):
    task = ABXpy.task.Task(item_file, 'c0', 'c1', 'c2')
    assert task.stats['nb_blocks'] == 8
    assert task.stats['nb_triplets'] == 8
    assert task.stats['nb_by_levels'] == 2

    task.generate_triplets(max_samples=task.stats['nb_triplets'])

    with h5py.File(item_file.replace('.item', '.abx'), 'r') as f:
        triplets = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0], [3, 2, 1]])
        assert tables_equivalent(triplets, get_triplets(f, '0'))
        assert tables_equivalent(triplets, get_triplets(f, '1'))

        pairs = {2, 6, 7, 3, 8, 12, 13, 9}
        assert pairs == set(get_pairs(f, '0')[:, 0])
        assert pairs == set(get_pairs(f, '1')[:, 0])


def test_sample_bad(item_file):
    task = ABXpy.task.Task(item_file, 'c0', 'c1', 'c2')

    with pytest.raises(AssertionError):
        task.generate_triplets(max_samples=0)

    with pytest.raises(AssertionError):
        task.generate_triplets(max_samples=0.1)


def create_unbalanced_items(name):
    items = ['#item #label']
    n = 0
    decode = {}
    for _ in range(5):
        items.append('i{} 1'.format(n))
        decode[n] = '1'
        n += 1
    for _ in range(3):
        items.append('i{} 2'.format(n))
        decode[n] = '2'
        n += 1
    for _ in range(2):
        items.append('i{} 3'.format(n))
        decode[n] = '3'
        n += 1
    open(name, 'w').write('\n'.join(items))

    return decode


def test_sample_all(item_file):
    decode = create_unbalanced_items(item_file)
    task = ABXpy.task.Task(item_file, 'label')

    task.generate_triplets()
    with h5py.File(item_file.replace('.item', '.abx'), 'r') as f:
        triplets = f['triplets']['data'][...]
        count = collections.Counter(
            tuple(decode[k] for k in t) for t in triplets)

        assert count == collections.Counter(
            {('1', '2', '1'): 60,
             ('1', '3', '1'): 40,
             ('2', '1', '2'): 30,
             ('2', '3', '2'): 12,
             ('3', '1', '3'): 10,
             ('3', '2', '3'): 6})


@pytest.mark.parametrize(
    'max_samples, ntriplets',
    [(10e6, 158), (10, 96), (5, 50), (2, 20), (1, 10)])
def test_sample_10(item_file, max_samples, ntriplets):
    create_unbalanced_items(item_file)
    task = ABXpy.task.Task(item_file, 'label')

    task.generate_triplets(max_samples=max_samples)
    with h5py.File(item_file.replace('.item', '.abx'), 'r') as f:
        triplets = f['triplets']['data'][...]

        # thsi assert is "hand-made" and we need a more rigorous test
        # of sampling
        assert len(triplets) == ntriplets
