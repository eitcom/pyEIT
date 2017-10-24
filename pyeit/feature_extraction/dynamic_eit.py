# pylint: disable=no-member, invalid-name
# pylint: disable=too-many-arguments, too-many-locals
"""
Dynamic EIT imaging and information retrieval from EIT images
"""
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from pyeit.eit import bp
from pyeit.eit import jac


class DynamicEIT(object):
    """ dynamic eit imaging """

    def __init__(self, mesh=None, el_pos=None, parser='et3',
                 solver='jac', p=0.20, lamb=0.001):
        """ initialize """
        if solver == 'jac':
            dyna_eit = jac.JAC(mesh, el_pos, perm=1., parser=parser)
            dyna_eit.setup(p=p, lamb=lamb, method='kotre')
        else:
            # default: 'bp'
            dyna_eit = bp.BP(mesh, el_pos, parser='fmmu', step=1)
            dyna_eit.setup(weight='simple')

        self.dyna_eit = dyna_eit

    @staticmethod
    def normalize(v, ref):
        """ normalize according to ref frame """
        raise NotImplementedError

    def map(self, v):
        """ map boundary voltages to EIT images """
        raise NotImplementedError
