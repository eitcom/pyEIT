from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np

import pyeit.eit.protocol as protocol
from pyeit.eit.fem import EITForward
from pyeit.mesh import create, set_perm
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
import random

if __name__ == "__main__":
    # Mesh shape is specified with fd parameter in the instantiation, e.g:
    # from pyeit.mesh.shape import thorax
    # mesh_obj, el_pos = create(n_el, h0=0.05, fd=thorax)  # Default : fd=circle
    n_el = 16  # test fem_vectorize
    mesh_obj = create(n_el, h0=0.04)

    # extract node, element, perm from mesh_obj
    xx, yy = mesh_obj.node[:, 0], mesh_obj.node[:, 1]
    tri = mesh_obj.element

    for i in range(0,10):
        r1 = random.uniform(-0.7,0.7)
        r2 = random.uniform(-0.7,0.7)
        r3 = random.uniform(0.1,0.2)
        r4 = random.uniform(-0.8,0.8)
        r5 = random.uniform(-0.8,0.8)
        r6 = random.uniform(0.1,0.2)

        r_perm_1 = random.uniform(1.2,3.5)
        # set anomaly (altering the permittivity in the mesh)
        anomaly = [
            PyEITAnomaly_Circle(center=[r1, r2], r=r3, perm=r_perm_1) 
           # PyEITAnomaly_Circle(center=[-r4, -r5], r=r6, perm=1.2),
        ]

        # background changed to values other than 1.0 requires more iterations
        mesh_new = set_perm(mesh_obj, anomaly=anomaly, background=1.0)
        perm = mesh_new.perm    

        # calculate simulated data
        protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=2, parser_meas="std")
        fwd = EITForward(mesh_obj, protocol_obj)
        v1 = fwd.solve_eit(perm=mesh_new.perm)  # v1 is ML input, and perm is ML output

        # plot
        fig, ax = plt.subplots(figsize=(9, 6))
        im = ax.tripcolor(xx, yy, tri, np.real(perm),  edgecolors="k", cmap="Reds", vmin=0, vmax=3.5)

        # Plot electrode positions
        # for el in mesh_obj.el_pos:
        #     ax.plot(xx[el], yy[el], "ro")

        ax.axis("equal")
        ax.set_title(r"Conductivities")
        fig.colorbar(im)
        plt.show(block=True)