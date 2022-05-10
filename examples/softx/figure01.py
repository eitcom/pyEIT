# coding: utf-8
# pylint: disable=invalid-name
""" Figure01 for softx """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function
import matplotlib.pyplot as plt

from pyeit.mesh import create
from pyeit.mesh import layer_circle

n_el = 16  # nb of electrodes
mesh0 = create(n_el)
el_pos0 = mesh0.el_pos
mesh1 = layer_circle(n_el, n_fan=8, n_layer=8)
el_pos1 = mesh1.el_pos

fig = plt.figure(figsize=(9, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# plot mesh0: distmesh
p = mesh0.node
t = mesh0.element
ax1.triplot(p[:, 0], p[:, 1], t, lw=1)
ax1.plot(p[el_pos0, 0], p[el_pos0, 1], "ro")
ax1.set_aspect("equal")
ax1.set_xlim([-1.5, 1.5])
ax1.set_ylim([-1.1, 1.1])
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title("distmesh2d")

# plot mesh1: layer circle
p = mesh1.node
t = mesh1.element
ax2.triplot(p[:, 0], p[:, 1], t, lw=1)
ax2.plot(p[el_pos1, 0], p[el_pos1, 1], "ro")
ax2.set_aspect("equal")
ax2.set_xlim([-1.5, 1.5])
ax2.set_ylim([-1.1, 1.1])
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title("layered circle")

fig.tight_layout()
fig.subplots_adjust(top=0.975, bottom=0.015)
# fig.set_size_inches(1, 1)
fig.savefig("figure01.pdf", dpi=300)
plt.show()
