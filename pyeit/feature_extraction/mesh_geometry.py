# pylint: disable=no-member, invalid-name, no-name-in-module
# pylint: disable=too-many-arguments, too-many-locals
"""using the geometry of mesh to segment the EIT images"""
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

import numpy as np
from numpy.linalg import eig, inv
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

from pkg_resources import resource_filename
from pyeit.io import mes
from pyeit.mesh import PyEITMesh


class SimpleMeshGeometry:
    """
    extract segments from meshes using:
    5-13 as the line for left and right hemisphere
    1-9 as the line for up and down

    Returns
    -------
    left, right, upper, down,
    upper-left, upper-right, down-left, down-right
    """

    # constructor
    def __init__(self, mesh: PyEITMesh, method="element"):
        """
        Parameters
        ----------
        mesh : dictionary
            'element', 'node', 'perm'
        method : string
            'element', 'node'
        """
        if method not in ["element", "node"]:
            raise TypeError("method do not recognized.")
        if method == "element":
            # find the center of elements
            self.ts = mesh.elem_centers
        else:
            self.ts = mesh.node
        el_pos = mesh.el_pos

        # vertical cut (from 5->13)
        pts = mesh.node
        vert = np.array([pts[el_pos[12]], pts[el_pos[4]]])
        self.vert_vec = vert[1] - vert[0]

        # horizonal cut (from 1->9)
        horz = np.array([pts[el_pos[0]], pts[el_pos[8]]])
        self.horz_vec = horz[1] - horz[0]

        # points vectors
        self.pv_vec = self.ts - vert[0]
        self.pv_vec = self.pv_vec.transpose()
        self.ph_vec = self.ts - horz[0]
        self.ph_vec = self.ph_vec.transpose()

    def left(self):
        """
        extract left coordinates
        note: bmp's right
        """
        return self._line_side(self.vert_vec, self.pv_vec)

    def right(self):
        """
        extract right coordinates
        note : bmp's left
        """
        return np.logical_not(self.left())

    def upper(self):
        """
        extract top coordinates
        note : bmp's bottom
        """
        return self._line_side(self.horz_vec, self.ph_vec)

    def down(self):
        """
        extract bottom coordinates
        note : bmp's top
        """
        return np.logical_not(self.upper())

    def upper_left(self):
        """upper left coordinates"""
        return np.logical_and(self.left(), self.upper())

    def upper_right(self):
        """upper right coordinates"""
        return np.logical_and(self.right(), self.upper())

    def down_left(self):
        """down left coordinates"""
        return np.logical_and(self.left(), self.down())

    def down_right(self):
        """down right coordinates"""
        return np.logical_and(self.right(), self.down())

    @staticmethod
    def _line_side(vline, vpoint):
        """
        Parameters
        ----------
        vline : array
            2x1 array, line[0] is the start point while line [1] is the end
            vline is line[1] - line[0]
        vpoint : array
            2x1 [x, y] coordinates, vpoint is point - line[0]

        find whether a pts is on the left of a line
        """
        proj = vline[0] * vpoint[1] - vline[1] * vpoint[0]
        return proj >= 0


class FitEllipse:
    """
    find the enclosing ellipse of a set of points

    see:
    [1] http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
    [2] https://www.cs.cornell.edu/cv/OtherPdf/Ellipse.pdf
    """

    def __init__(self):
        """initialize mesh"""
        self.a = np.zeros(6)
        # self.pts = mesh['node']
        # self.tri = mesh['element']
        # self.hull_points = self.convex_hull_points(self.pts)

    def fit(self, pts):
        """
        find the enclosing ellipse of data points
        """
        hpts = self.convex_hull_points(pts)
        x, y = hpts[:, 0], hpts[:, 1]
        a = self.fit_ellipse(x, y)
        # x_cent, y_cent
        cx, cy = self.ellipse_center(a)
        # semi major, semi minor
        ca, cb = self.ellipse_axis_length(a)
        # offset angle (radius)
        cangle = self.ellipse_angle_of_rotation(a)
        return np.array([cx, cy, ca, cb, cangle])

    @staticmethod
    def convex_hull_points(pts):
        """
        finding the convex hull (points) of data
        """
        cv = ConvexHull(pts)
        hull_nodes = cv.vertices
        return pts[hull_nodes, :]

    @staticmethod
    def fit_ellipse(x, y):
        """
        fit ellipse using least squares
        """
        x = x[:, np.newaxis]
        y = y[:, np.newaxis]
        D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
        S = np.dot(D.T, D)
        C = np.zeros([6, 6])
        C[0, 2] = C[2, 0] = 2
        C[1, 1] = -1
        E, V = eig(np.dot(inv(S), C))
        n = np.argmax(np.abs(E))
        a = V[:, n]
        return a

    @staticmethod
    def ellipse_center(a):
        """
        Returns
        -------
        x_center, y_center
        """
        b, c, d, f, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[0]
        num = b * b - a * c
        x0 = (c * d - b * f) / num
        y0 = (a * f - b * d) / num
        return np.array([x0, y0])

    @staticmethod
    def ellipse_axis_length(a):
        """
        Returns
        -------
        semiminor, semimajor axis length
        """
        b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
        up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
        down1 = (b * b - a * c) * (
            (c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a)
        )
        down2 = (b * b - a * c) * (
            (a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a)
        )
        res1 = np.sqrt(up / down1)
        res2 = np.sqrt(up / down2)
        return np.array([res1, res2])

    @staticmethod
    def ellipse_angle_of_rotation(a):
        """
        Returns
        -------
        ellipse angle in radius
        """
        b, c, a = a[1] / 2, a[2], a[0]
        return 0.5 * np.arctan(2 * b / (a - c))

    @staticmethod
    def ellipse_angle_of_rotation2(a):
        """
        see:
        [1] http://mathworld.wolfram.com/Ellipse.html
        """
        b, c, a = a[1] / 2, a[2], a[0]
        if b == 0:
            if a > c:
                phi = 0
            else:
                phi = np.pi / 2
        else:
            if a > c:
                phi = np.arctan(2 * b / (a - c)) / 2
            else:
                phi = np.pi / 2 + np.arctan(2 * b / (a - c)) / 2

        return phi


def ellipse_points(x_cent=0, y_cent=0, semimaj=1, semimin=1, phi=0, theta_num=1e3):
    """
    see:
    https://casper.berkeley.edu/astrobaki/index.php/Plotting_Ellipses_in_Python

    Create ellipse points.
    The function creates a 2D ellipse in polar coordinates
    then transforms to cartesian coordinates.
    (un-implemented)
    It can take a covariance matrix and plot contours from it.

    x_cent : float
        X coordinate center

    y_cent : float
        Y coordinate center

    semimaj : float
        length of semimajor axis
        always taken to be some phi (-90<phi<90 deg) from
        positive x-axis!

    semimin : float
        length of semiminor axis

    phi : float
        angle in radians of semimajor axis above positive x axis

    theta_num : int
        Number of points to sample along ellipse from 0-2pi

    """
    # Generate data for ellipse structure
    theta = np.linspace(0, 2 * np.pi, int(theta_num))
    r = 1 / np.sqrt((np.cos(theta)) ** 2 + (np.sin(theta)) ** 2)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    data = np.array([x, y])
    S = np.array([[semimaj, 0], [0, semimin]])
    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    T = np.dot(R, S)
    data = np.dot(T, data)
    data[0] += x_cent
    data[1] += y_cent

    return data.transpose()


if __name__ == "__main__":
    # load package mesh data for pyEIT (data/*.mes)
    mstr = resource_filename("eitmesh", "data/DLS2.mes")
    print(mstr)

    # load mesh
    mesh = mes.load(mstr)
    pts = mesh.node
    tri = mesh.element
    el_pos = mesh.el_pos
    x, y = pts[:, 0], pts[:, 1]

    # 1. demo using ellipse fit
    # load shape
    eobj = FitEllipse()
    hpts = eobj.convex_hull_points(pts)
    cvx = hpts[:, 0]
    cvy = hpts[:, 1]

    # ellipse fit
    v = eobj.fit(pts)
    x_cent, y_cent, semimaj, semimin, phi = v
    epts = ellipse_points(x_cent, y_cent, semimaj, semimin, phi)

    # plot mesh
    fig, ax = plt.subplots(figsize=(9, 6))
    mesh_image = mstr.replace(".mes", ".bmp")
    im = plt.imread(mesh_image)
    ax.imshow(im)
    ax.triplot(x, y, tri, "g-", lw=1.0, alpha=0.5)
    ax.set_aspect("equal")
    # plot convex hulls of mesh
    ax.plot(cvx, cvy, "go")
    ax.plot(epts[:, 0], epts[:, 1], "b-", lw=2.0)
    # plot electrodes and its numbering
    ax.plot(pts[el_pos, 0], pts[el_pos, 1], "bo")
    for i, e in enumerate(el_pos):
        ax.text(pts[e, 0], pts[e, 1], str(i + 1), color="r")

    # 2. demo using simple fit
    mg = SimpleMeshGeometry(mesh)
    perm = np.zeros(tri.shape[0])
    fig, ax = plt.subplots(figsize=(9, 6))
    mesh_image = mstr.replace(".mes", ".bmp")
    im = plt.imread(mesh_image)
    ax.imshow(im)
    # plot regions
    perm[mg.left()] = 1.0
    perm[mg.down_right()] = 2.0
    img = ax.tripcolor(pts[:, 0], pts[:, 1], tri, perm, shading="flat", alpha=0.50)
    # plot electrodes and its numbering
    ax.plot(pts[el_pos, 0], pts[el_pos, 1], "bo")
    ax.set_title("left = 1.0, down right = 2.0")
    for i, e in enumerate(el_pos):
        ax.text(pts[e, 0], pts[e, 1], str(i + 1), color="r", size=12)
    fig.colorbar(img)

    plt.show()
