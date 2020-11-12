# pylint: disable=no-member, invalid-name
"""
open/view .mes file (a binary mesh structure)

The mesh structure in mes was developed by FMMU EIT group (Bin Yang, et al.)
Please Cite the following paper if you are using .mes in your research:
    Yang, Bin, et al. "Comparison of electrical impedance tomography and
    intracranial pressure during dehydration treatment of cerebral edema."
    NeuroImage: Clinical 23 (2019): 101909.
"""
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
import ctypes
import struct
import numpy as np
import matplotlib.pyplot as plt
from pkg_resources import resource_filename


def load(fstr, mirror=False):
    """
    Parameters
    ----------
    mirror: mirror left and right electrodes

    return mesh and el_pos
    """

    # note: seek is global position,
    # fh can be advanced using read in sub-function
    with open(fstr, "rb") as fh:

        # 0. extract BMP
        # (offset=bmp_size)
        bmp_size = get_bmp_size(fh)
        bmp = bytearray(fh.read(bmp_size))
        save_bmp(fstr, bmp)

        # 1. extract mesh [104 Bytes / structure]
        # ne = tri.shape[0]
        # (offset = 4 + ne*104)
        tri, perm = extract_element(fh)

        # 2. extract nodes [20 Bytes / structure]
        # nn = pts.shape[0]
        # (offset=4 + nn*20)
        pts = extract_node(fh)

        # 3. extract electrodes
        el_pos = extract_el(fh)

    if mirror:
        ne = np.size(el_pos)
        ne_start = np.int(ne / 2)
        el_index = np.mod(np.arange(ne_start, -ne_start, -1), ne)
        el_pos = el_pos[el_index]

    mesh = {"node": pts, "element": tri, "perm": perm}
    return mesh, el_pos


def get_bmp_size(fh):
    """get the size of bmp segment"""
    # size of BMP is stored in a UNSIGNED LONG LONG at the end of .mes
    nff = ctypes.sizeof(ctypes.c_int64)
    # seek backwards from the end (2) of the file
    fh.seek(-nff, 2)
    # unsigned long long is 'Q'
    bmp_size = struct.unpack("Q", fh.read(nff))[0]
    # fh is a file ID,
    # it can be modified via read within subroutine, rewind!
    fh.seek(0)

    return bmp_size


def save_bmp(fstr, bmp):
    """save bmp segment to file"""
    # automatically infer file name from 'fstr'
    bmp_file = fstr.replace(".mes", ".bmp")
    with open(bmp_file, "wb") as fh:
        fh.write(bmp)


def extract_element(fh):
    """
    extract element segment

    Notes
    -----
    structure of each element:
    {
        int node1;       // node-1 (global numbering)
        int node2;       // node-2 (global numbering)
        int node3;       // node-3 (global numbering)
        int gENum;       // element (global numbering)
        double alpha;    // conductive (sigma)
        double Ae[3][3]; // sensitivity matrix (local) see. Tadakuni murai
        double deltae;   // area of this element
    }
    = 104 Bytes
    """
    # number of element
    nff = ctypes.sizeof(ctypes.c_int)
    ne = struct.unpack("i", fh.read(nff))[0]
    tri = np.zeros((ne, 3), dtype="int")
    perm = np.zeros(ne, dtype="double")

    for _ in range(ne):
        d = np.array(struct.unpack("4i10dd", fh.read(104)))
        # global element number
        ge_num = int(d[3])
        tri[ge_num, :] = d[:3]
        perm[ge_num] = d[4]

    return tri, perm


def extract_node(fh):
    """
    extract node structure

    Notes
    -----
    structure of each node
    {
        double x; // x-coordinate
        double y; // y-coordinate
        int gNum; // node (global numbering)
    }
    = 20 Bytes
    """
    # number of nodes
    nff = ctypes.sizeof(ctypes.c_int)
    nn = struct.unpack("i", fh.read(nff))[0]
    pts = np.zeros((nn, 2), dtype="double")

    for _ in range(nn):
        d = np.array(struct.unpack("2di", fh.read(20)))
        # global node number
        gn_num = int(d[2])
        # offset to overlay on .bmp file
        # x_new = x+8, y_new = -y-8 (yang bin)
        # x_new = x-8, y_new = -y-8 (lby)
        pts[gn_num, :] = [d[0] - 4.0, -d[1] - 4.0]

    return pts


def extract_el(fh):
    """extract the locations of electrodes"""
    # how many electrodes in .mes
    nff = ctypes.sizeof(ctypes.c_int)
    ne = struct.unpack("i", fh.read(nff))[0]
    # read all at once
    el_pos = np.array(struct.unpack(ne * "i", fh.read(ne * 4)))

    return el_pos


if __name__ == "__main__":
    # How to load and use a .mes file (github.com/liubenyuan/pyeit-models)
    mstr = resource_filename("pyeit-models", "data/model/DLS2.mes")
    mesh_obj, el_pos = load(fstr=mstr)

    # print the size
    e, pts = mesh_obj["element"], mesh_obj["node"]
    # print('tri size = (%d, %d)' % e.shape)
    # print('pts size = (%d, %d)' % pts.shape)

    # show mesh
    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.triplot(pts[:, 0], pts[:, 1], e)
    ax.plot(pts[el_pos, 0], pts[el_pos, 1], "ro")
    for i, el in enumerate(el_pos):
        ax.text(pts[el, 0], pts[el, 1], str(i + 1), color="r")
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # bmp and mesh overlay
    fig, ax = plt.subplots(figsize=(6, 6))
    imstr = mstr.replace("mes", "bmp")
    im = plt.imread(imstr)
    ax.imshow(im)
    ax.set_aspect("equal")

    # the plot will automatically align with an overlay image
    ax.triplot(pts[:, 0], pts[:, 1], e)
    ax.plot(pts[el_pos, 0], pts[el_pos, 1], "ro")
    for i, el in enumerate(el_pos):
        ax.text(pts[el, 0], pts[el, 1], str(i + 1), color="r")
    ax.axis("off")
    plt.show()
