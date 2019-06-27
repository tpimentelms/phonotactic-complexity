import numpy as np


def paired_permutation_test(art, norm, verbose=True):
    diffs = [art[i] - norm[i] for i in range(len(art))]
    mu = np.abs(np.mean(diffs))

    n = 0
    for i in range(2**len(art)):
        permut = bin(i)[2:]
        permut = [0] * (len(art) - len(permut)) + [int(y) for y in permut]
        mu_new = np.abs(np.mean([x if j else -x for x, j in zip(diffs, permut)]))
        if mu_new >= mu:
            n += 1

    if verbose:
        print(diffs)
        print(n, n / 2**len(art))
    return n / 2**len(art)


def paired_permutation_test_partial(art, norm, ntests=10000, verbose=True):
    diffs = [art[i] - norm[i] for i in range(len(art))]
    mu = np.abs(np.mean(diffs))

    n = 0
    for _ in range(ntests):
        permut = np.random.uniform(0, 1, size=(len(art))) > 0.5
        permut = [0] * (len(art) - len(permut)) + [int(y) for y in permut]
        mu_new = np.abs(np.mean([x if j else -x for x, j in zip(diffs, permut)]))
        if mu_new >= mu:
            n += 1

    if verbose:
        print(n, n / ntests)
    return n / ntests
