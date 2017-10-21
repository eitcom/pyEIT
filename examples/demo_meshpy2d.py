# coding: utf-8
# pylint: disable=invalid-name
"""
Demo for MeshPy 2D. MeshPy is required before running this demo.

MeshPy can be installed under linux using conda or pip.
In windows, there is no working version that is shipped with a precompiled
dll. The user is recommended download MeshPy from:
http://www.lfd.uci.edu/~gohlke/pythonlibs/#meshpy
and install it using (current version at 2017/10/20 is 2016.1.2),
$ pip install MeshPy‑2016.1.2‑cp36‑cp36m‑win_amd64.whl

MeshPy is extremely fast and versatile than distmesh (in pyEIT).
It is using Triangle by J. Shewchuk and TetGen by Hang Si. See:
https://mathema.tician.de/software/meshpy/
or
https://github.com/inducer/meshpy
for more details.
"""
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

import matplotlib.pyplot as plt
from pyeit.mesh.meshpy.build import create
from pyeit.mesh.meshpy import shape


def example1():
    """ unit circle mesh """

    # build triangle
    # curve = {'disc',
    #          'disc_anomaly',
    #          'throx',
    #          'throx_anomaly'}
    mesh_obj, el_pos = create(16, max_area=0.004, curve=shape.throx,
                              refine=True)
    p = mesh_obj['node']
    t = mesh_obj['element']

    # plot
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.triplot(p[:, 0], p[:, 1], t)
    ax.plot(p[el_pos, 0], p[el_pos, 1], 'ro')
    for (i, e) in enumerate(el_pos):
        ax.text(p[e, 0], p[e, 1], str(i+1))
    ax.set_aspect('equal')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    plt.show()


if __name__ == "__main__":
    example1()
