# coding: utf-8
# pylint: disable=invalid-name
""" util functions for 2D EIT """
# author: benyuan liu
from __future__ import division, absolute_import, print_function

import numpy as np


def eit_scan_lines(ne=16, dist=1):
    """
    generate scan matrix

    Parameters
    ----------
    ne : int
        number of electrodes
    dist  : int
        distance between A and B (default=1)

    Returns
    -------
    ex_mat : NDArray
        excitation matrix

    Notes
    -----
    in the scan of EIT (or excitation matrix), we use 4-electrodes
    mode, where A, B are used as positive and negative excitation
    electrodes and M, N are used as voltage measurements

         1 (A) for positive current injection,
        -1 (B) for negative current sink

    dist is the distance (number of electrodes) of A to B
    in 'adjacent' mode, dist=1, in 'apposition' mode, dist=ne/2

    Examples
    --------
    if excitation_mode=='neighbor':
        ex_mat = eit_scan_lines(ne)
    elif excitation_mode=='apposition':
        ex_mat = eit_scan_lines(ne, ne/2)
    """
    # A: diagonal
    ex_pos = np.eye(ne)
    # B: rotate right by dist
    ex_neg = -1 * np.roll(ex_pos, dist, axis=1)
    ex = ex_pos + ex_neg

    return ex


if __name__ == "__main__":
    m = eit_scan_lines()
    print(m)
