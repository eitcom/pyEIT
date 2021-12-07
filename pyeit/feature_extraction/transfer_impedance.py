# pylint: disable=invalid-name, too-many-locals
"""static features of R values measured using EIT system"""
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

import numpy as np


def ati(x):
    """
    averaged total impedance, unit (mV),
    original / 192.0
    [experimental] new / 16.0

    Notes
    -----
    if I=1mA, then ati returns Ohms
    """
    # implement old behavior of numpy.nansum
    if np.isnan(x).any():
        v = np.nan
    else:
        v = np.sum(np.abs(x)) / 192.0

    return v


def ati_df(x):
    """ati of DataFrame"""
    return x.abs().sum(skipna=False) / 192.0


def fmmu_index(n_el=16, dist=8, step=1):
    """
    generate FMMU patterns and left, right index

    n_el = 16
    dist = 8 # opposition
    step = 1 # adjacent measures
    """
    # [0, ..., 15]
    left_el = [13, 14, 15, 0, 1, 2, 3, 4]
    right_el = [5, 6, 7, 8, 9, 10, 11, 12]

    m_array = []
    for a in range(n_el):
        b = (a + dist) % n_el
        for j in range(a, a + n_el):
            m = j % n_el
            n = (j + step) % n_el
            if not (m in [a, b] or n in [a, b]):
                diff_pair = [n, m]  # v_n - v_m
                m_array.append(diff_pair)

    m_array = np.array(m_array)
    N = m_array.shape[0]
    left_sel = np.zeros(N, dtype=np.bool)  # 192
    right_sel = np.zeros(N, dtype=np.bool)
    for i, nm in enumerate(m_array):
        n = nm[0]
        if n in left_el:
            left_sel[i] = True
        if n in right_el:
            right_sel[i] = True

    return left_sel, right_sel


def ati_lr(x, sel):
    """extract ATI left, right"""
    x_sel = np.nanmean(np.abs(x[sel]))

    return x_sel


def rchannel(x, offset=0):
    """
    calculate R (voltages) of two electrodes counted from the
    excitation electrode with a fixed number (offset)
    """
    N = 12  # measurements within a excitation pattern
    c = np.mod(offset, N)
    x = x[:, c:N:]
    return np.abs(x).sum(axis=1) / float(N)
