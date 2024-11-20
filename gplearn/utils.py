"""Utilities that are required by gplearn.

Most of these functions are slightly modified versions of some key utility
functions from scikit-learn that gplearn depends upon. They reside here in
order to maintain compatibility across different versions of scikit-learn.

"""

import numbers

import numpy as np
from joblib import cpu_count


import pandas as pd
import numpy as np

def calc_cross_sectional_xcr(xdf, ydf, option="pearson"):
    xdf = pd.DataFrame(xdf)
    ydf = pd.DataFrame(ydf)
    NonNanMask = np.where(xdf.mul(ydf).isnull().values, np.nan, 1)
    NonNanMaskDf = pd.DataFrame(index=xdf.index, columns=xdf.columns, data=NonNanMask)
    xdf = xdf.fillna(0).mul(NonNanMaskDf)
    ydf = ydf.fillna(0).mul(NonNanMaskDf)
    Cntx = xdf.count(axis=1).apply(lambda x: x if x > 0 else np.nan)
    Cnt = ydf.count(axis=1).apply(lambda x: x if x > 0 else np.nan)
    if option == 'pearson':
        pass
    elif option =='spearman':
        xdf = xdf.rank(axis=1)
        ydf = ydf.rank(axis=1)
    else:
        raise ValueError(f'Unspecified option=[option]')
    SumXY = xdf.mul(ydf).sum(axis=1)
    SumX = xdf.sum(axis=1)
    SumY = ydf.sum(axis=1)
    Sumx2 = xdf.mul(xdf).sum(axis=1)
    SumY2 = ydf.mul(ydf).sum(axis=1)
    P1 = Cnt * SumXY - SumX * SumY
    P2 = ((Cnt * Sumx2 - SumX * SumX) * (Cnt * SumY2 - SumY * SumY)).apply(lambda x: np.sqrt(x) if x > 1e-10 else np.nan)
    Xcr = P1 / P2
    ErrList = np.argwhere(np.abs(Xcr.values) > 1 + 1e-6)
    for idx in ErrList:
        print(f"***ERR: idx={idx} Xcr={Xcr.values[idx]}, XCount={Cntx.values[idx]}, YCount={Cnt.values[idx]}")
    assert len(ErrList) == 0, f'Xcr.abs>1 on {len(ErrList)} samples.'
    return Xcr


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def _get_n_jobs(n_jobs):
    """Get number of jobs for the computation.

    This function reimplements the logic of joblib to determine the actual
    number of jobs depending on the cpu count. If -1 all CPUs are used.
    If 1 is given, no parallel computing code is used at all, which is useful
    for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
    Thus for n_jobs = -2, all CPUs but one are used.

    Parameters
    ----------
    n_jobs : int
        Number of jobs stated in joblib convention.

    Returns
    -------
    n_jobs : int
        The actual number of jobs as positive integer.

    """
    if n_jobs < 0:
        return max(cpu_count() + 1 + n_jobs, 1)
    elif n_jobs == 0:
        raise ValueError('Parameter n_jobs == 0 has no meaning.')
    else:
        return n_jobs


def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(_get_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = (n_estimators // n_jobs) * np.ones(n_jobs,
                                                              dtype=int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()
