# coding: utf-8
# pylint: disable=invalid-name, no-member
""" implement quality assesment functions for distmesh """
from __future__ import absolute_import


def fstats(p, t):
    """ print mesh or tetrahedral status

    Parameters
    ----------
    p : array_like
        coordinates of nodes
    t : array_like
        simplices of the connectives of elements

    Notes
    -----
    a simple function for illustration purpose only.
    print the status (size) of nodes and elements
    """
    print('mesh status:')
    print('%d nodes, %d elements' % (len(p), len(t)))
