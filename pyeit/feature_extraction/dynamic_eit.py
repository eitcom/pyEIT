# pylint: disable=no-member, invalid-name
# pylint: disable=too-many-arguments, too-many-locals
"""
Dynamic EIT imaging and information retrieval from EIT images
"""
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from pyeit.eit import bp
from pyeit.eit import jac


class DynamicEIT:
    """dynamic eit imaging"""

    def __init__(
        self, mesh=None, el_pos=None, parser="fmmu", algo="jac", p=0.20, lamb=0.001
    ):
        """initialize"""
        if algo == "jac":
            solver = jac.JAC(mesh, el_pos, perm=1.0, parser=parser)
            solver.setup(p=p, lamb=lamb, method="kotre")
        else:
            # default: 'bp'
            solver = bp.BP(mesh, el_pos, parser="fmmu", step=1)
            solver.setup(weight="simple")

        self.solver = solver

    def normalize(self, v1, v0):
        """normalize according to ref frame"""
        raise NotImplementedError

    def map(self, v):
        """map boundary voltages to EIT images"""
        raise NotImplementedError
