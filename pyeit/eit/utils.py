# coding: utf-8
# pylint: disable=invalid-name
"""
util functions for 2D EIT
1. generate stimulation lines/patterns
"""
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

import numpy as np


def eit_scan_lines(n_el: int = 16, dist: int = 1) -> np.ndarray:
    """
    Generate scan matrix, `ex_mat` ( or excitation pattern), see notes

    Parameters
    ----------
    n_el : int, optional
        number of electrodes, by default 16
    dist : int, optional
        distance (number of electrodes) of A to B, by default 1
        For 'adjacent'- or 'neighbore'-mode (default) use `1` , and
        for 'apposition'-mode use `n_el/2` (see Examples).

    Returns
    -------
    np.ndarray
        stimulation matrix; shape (n_exc, 2)

    Notes
    -----
        - in the scan of EIT (or stimulation matrix), we use 4-electrodes
        mode, where A, B are used as positive and negative stimulation
        electrodes and M, N are used as voltage measurements.
        - `1` (A) for positive current injection, `-1` (B) for negative current
        sink

    Examples
    --------
        n_el=16
        if mode=='neighbore':
            ex_mat = eit_scan_lines(n_el=n_el)
        elif mode=='apposition':
            ex_mat = eit_scan_lines(dist=n_el/2)

    WARNING
    -------
        `ex_mat` is a local index, where it is ranged from 0...15, within the
        range of the number of electrodes. In FEM applications, you should
        convert `ex_mat` to global index using the (global) `el_pos` parameters.
    """
    return np.array([[i, np.mod(i + dist, n_el)] for i in range(n_el)])


if __name__ == "__main__":
    m = eit_scan_lines(dist=8)
    print(m)
