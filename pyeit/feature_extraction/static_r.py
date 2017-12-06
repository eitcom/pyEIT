# pylint: disable=invalid-name
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
    v = np.abs(x).sum() / 192.0
    return v


def rchannel(x, offset=0):
    """
    calculate R (voltages) of two electrodes counted from the
    excitation electrode with a fixed number (offset)
    """
    N = 12  # measurements within a excitation pattern
    c = np.mod(offset, N)
    x = x[:, c:N:]
    return np.abs(x).sum(axis=1) / float(N)
