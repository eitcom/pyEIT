# coding: utf-8
# pylint: disable=invalid-name, no-member, arguments-differ
""" EIT protocol """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import absolute_import, division, print_function, annotations

from dataclasses import dataclass
from typing import Union, List, Tuple

import numpy as np


@dataclass
class PyEITProtocol:
    """
    EIT Protocol buid-in protocol object

    Parameters
    ----------
    ex_mat: np.ndarray
        excitation matrix (pairwise)
    meas_mat: np.ndarray
        measurement matrix (differential pairs), support inhomogeneous number of measurements per excitation pair.
    keep_ba: np.ndarray
        boolean array index for keeping measurements
    """

    ex_mat: np.ndarray
    meas_mat: np.ndarray
    keep_ba: np.ndarray

    def __post_init__(self) -> None:
        """Checking of the inputs"""
        self.ex_mat = self._check_ex_mat(self.ex_mat)
        self.meas_mat = self._check_meas_mat(self.meas_mat)
        self.keep_ba = self._check_keep_mat(self.keep_ba)

    def _check_ex_mat(self, ex_mat: np.ndarray) -> np.ndarray:
        """
        Check/init stimulation

        Parameters
        ----------
        ex_mat : np.ndarray
            stimulation/excitation matrix, of shape (n_exc, 2).
            If single stimulation (ex_line) is passed only a list of length 2
            and np.ndarray of size 2 will be treated.

        Returns
        -------
        np.ndarray
            stimulation matrix

        Raises
        ------
        TypeError
            Only accept, list of length 2, np.ndarray of size 2,
            or np.ndarray of shape (n_exc, 2)
        """
        if isinstance(ex_mat, list) and len(ex_mat) == 2:
            # case ex_line has been passed instead of ex_mat
            ex_mat = np.array([ex_mat]).reshape((1, 2))  # build a 2D array
        elif isinstance(ex_mat, np.ndarray) and ex_mat.size == 2:
            # case ex_line np.ndarray has been passed instead of ex_mat
            ex_mat = ex_mat.reshape((-1, 2))

        if not isinstance(ex_mat, np.ndarray):
            raise TypeError(f"Wrong type of {type(ex_mat)=}, expected an ndarray;")
        if ex_mat.ndim != 2 or ex_mat.shape[1] != 2:
            raise TypeError(f"Wrong shape of {ex_mat.shape=}, should be (n_exc, 2);")

        return ex_mat

    def _check_meas_mat(self, meas_mat: np.ndarray) -> np.ndarray:
        """
        Check measurement pattern

        Parameters
        ----------
        n_exc : int
            number of excitations/stimulations
        meas_pattern : np.ndarray, optional
           measurements pattern / subtract_row pairs [N, M]; shape (n_meas_tot, 3)

        Returns
        -------
        np.ndarray
            measurements pattern / subtract_row pairs [N, M]; shape (n_meas_tot, 3)

        Raises
        ------
        TypeError
            raised if meas_pattern is not a np.ndarray of shape (n_meas_tot, 3)
        """
        if not isinstance(meas_mat, np.ndarray):
            raise TypeError(f"Wrong type of {type(meas_mat)=}, expected an ndarray;")
        if meas_mat.ndim != 2 or meas_mat.shape[-1] != 3:
            raise TypeError(f"{meas_mat.shape=} must be (n_meas_tot, 3);")

        return meas_mat

    def _check_keep_mat(self, keep_ba: np.ndarray) -> np.ndarray:
        """check keep boolean array"""
        if not isinstance(keep_ba, np.ndarray):
            raise TypeError(f"Wrong type of {type(keep_ba)=}, expected an ndarray;")

        return keep_ba

    @property
    def n_exc(self) -> int:
        """
        Returns
        -------
        int
            number of excitation
        """
        return self.ex_mat.shape[0]

    @property
    def n_meas(self) -> int:
        """
        Returns
        -------
        int
            total amount of measurements (n_meas_tot)
        """
        return self.meas_mat.shape[0]

    @property
    def n_el(self) -> int:
        """
        Returns
        -------
        int
            infer the number of electrodes used in the excitation and measurements patterns,
            where the electrodes are numbered [0, n_el-1].
        """
        return (
            int(max(max(self.ex_mat.flatten()), max(self.meas_mat[:, :-1].flatten())))
            + 1
        )


def create(
    n_el: int = 16,
    dist_exc: Union[int, List[int]] = 1,
    step_meas: int = 1,
    parser_meas: Union[str, List[str]] = "std",
) -> PyEITProtocol:
    """
    Return an EIT protocol, comprising an excitation and a measuremnet pattern

    Parameters
    ----------
    n_el : int, optional
        number of total electrodes, by default 16
    dist_exc : Union[int, List[int]], optional
        distance (number of electrodes) of A to B, by default 1
        For 'adjacent'- or 'neighbore'-mode (default) use `1` , and
        for 'apposition'-mode use `n_el/2`. (see `build_exc_pattern`)
        if a list of integer is passed the excitation will bee stacked together.
    step_meas : int, optional
    measurement method (two adjacent electrodes are used for measuring), by default 1 (adjacent).
        (see `build_meas_pattern`)
    parser_meas : Union[str, List[str]], optional
        parsing the format of each frame in measurement/file, by default 'std'.
        (see `build_meas_pattern`)

    Returns
    -------
    PyEITProtocol
        EIT protocol object

    Raises
    ------
    TypeError
        if dist_exc is not list or an int
    """
    if isinstance(dist_exc, int):
        dist_exc = [dist_exc]

    if not isinstance(dist_exc, list):
        raise TypeError(f"{type(dist_exc)=} should be a List[int]")

    _ex_mat = [build_exc_pattern_std(n_el, dist) for dist in dist_exc]
    ex_mat = np.vstack(_ex_mat)

    meas_mat, keep_ba = build_meas_pattern_std(ex_mat, n_el, step_meas, parser_meas)
    return PyEITProtocol(ex_mat, meas_mat, keep_ba)


def build_meas_pattern_std(
    ex_mat: np.ndarray,
    n_el: int = 16,
    step: int = 1,
    parser: Union[str, List[str]] = "std",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the measurement pattern (subtract_row-voltage pairs [N, M]) for all excitations on boundary electrodes.
    The excitation index (exc_id) are also recorded for computing subtract_row_vectorized and smear_nd.

    we direct operate on measurements or Jacobian on electrodes,
    so, we can use LOCAL index in this module, do not require el_pos.

    Notes
    -----
    ABMN Model.
    A: current driving electrode,
    B: current sink,
    M, N: boundary electrodes, where v_diff = v_n - v_m.

    Parameters
    ----------
    ex_mat : np.ndarray
        Nx2 array, [positive electrode, negative electrode]. ; shape (n_exc, 2)
    n_el : int, optional
        number of total electrodes, by default 16
    step : int, optional
        measurement method (two adjacent electrodes are used for measuring), by default 1 (adjacent)
    parser : Union[str, List[str]], optional
        parsing the format of each frame in measurement/file, by default 'std'
        if parser contains 'fmmu', or 'rotate_meas' then data are trimmed,
        boundary voltage measurements are re-indexed and rotated,
        start from the positive stimulus electrode start index 'A'.
        if parser contains 'std', or 'no_rotate_meas' then data are trimmed,
        the start index (i) of boundary voltage measurements is always 0.
        if parser contains 'meas_current', the measurements on current carrying
        electrodes are allowed. Otherwise the measurements on current carrying
        electrodes are discarded (like 'no_meas_current' option in EIDORS3D).

    Returns
    -------
    diff_op: np.ndarray
        measurements pattern / subtract_row pairs, and the excitation indice;
        shape (n_meas_tot, 3), for each row, it represents [Ni, Mi, exc_id]
    keep_ba: np.ndarray
        (n_meas_tot,) boolean array
    """
    if not isinstance(parser, list):  # transform parser into list
        parser = [parser]
    meas_current = "meas_current" in parser
    fmmu_rotate = any(p in ("fmmu", "rotate_meas") for p in parser)

    diff_op, keep_ba = [], []
    for exc_id, exc_line in enumerate(ex_mat):
        a, b = exc_line[0], exc_line[1]
        i0 = a if fmmu_rotate else 0
        # build [[m, n, idx]_i] array
        m = (i0 + np.arange(n_el)) % n_el
        n = (m + step) % n_el
        idx = exc_id * np.ones(n_el)
        meas_pattern = np.vstack([n, m, idx]).T

        diff_keep = np.logical_and.reduce((m != a, m != b, n != a, n != b))
        keep_ba.append(diff_keep)
        if not meas_current:
            meas_pattern = meas_pattern[diff_keep]
        diff_op.append(meas_pattern.astype(int))

    return np.vstack(diff_op), np.array(keep_ba).ravel()


def build_exc_pattern_std(n_el: int = 16, dist: int = 1) -> np.ndarray:
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
            ex_mat = build_exc_pattern(n_el=n_el)
        elif mode=='apposition':
            ex_mat = build_exc_pattern(dist=n_el/2)

    WARNING
    -------
        `ex_mat` is a local index, where it is ranged from 0...15, within the
        range of the number of electrodes. In FEM applications, you should
        convert `ex_mat` to global index using the (global) `el_pos` parameters.
    """
    return np.array([[i, np.mod(i + dist, n_el)] for i in range(n_el)])
