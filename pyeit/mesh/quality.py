# coding: utf-8
# pylint: disable=invalid-name, no-member
""" implement quality assesment functions for distmesh """
from __future__ import absolute_import


def fstats(p, t):
    """ print mesh or tetrahedral status """
    print('%d nodes, %d elements' % (len(p), len(t)))
