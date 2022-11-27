# coding: utf-8
# pylint: disable=invalid-name, no-member, too-many-arguments
""" wrapper function of distmesh for EIT """
# Copyright (c) Benyuan Liu. All rights reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import absolute_import, division, print_function, annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Union, List, Any, Optional

import numpy as np

from . import shape
from .distmesh import build
from .mesh_circle import MeshCircle
from .shape import ball, circle
from .utils import check_order


@dataclass
class PyEITMesh:
    """
    Pyeit buid-in mesh object

    Parameters
    ----------
    node : np.ndarray
        node of the mesh of shape (n_nodes, 2), (n_nodes, 3)
    element : np.ndarray
        elements of the mesh of shape (n_elem, 3) for 2D mesh, (n_elem, 4) for 3D mesh
    perm : Union[int, float, np.ndarray], optional
        permittivity on elements; shape (n_elems,), by default `None`.
        If `None`, a uniform permittivity on elements with a value 1 will be generated.
        If perm is int or float, uniform permittivity on elements with value of perm will be generated.
    el_pos : np.ndarray
        node corresponding to each electrodes of shape (n_el, 1)
    ref_node : int
        reference node. ref_node should not be on electrodes, default 0.
    """

    node: np.ndarray
    element: np.ndarray
    perm: Union[int, float, complex, np.ndarray] = field(default_factory=lambda: 1.0)
    el_pos: np.ndarray = field(default_factory=lambda: np.arange(16))
    ref_node: int = field(default_factory=lambda: 0)

    def __post_init__(self) -> None:
        """Checking of the inputs"""
        self.element = self._check_element(self.element)
        self.node = self._check_node(self.node)
        # self.perm = self.get_valid_perm_array(self.perm)
        self.ref_node = self._check_ref_node(self.ref_node)

    def print_stats(self):
        """
        Print mesh or tetrahedral status

        Parameters
        ----------
        p: array_like
            coordinates of nodes (x, y) in 2D, (x, y, z) in 3D
        t: array_like
            connectives forming elements

        Notes
        -----
        a simple function for illustration purpose only.
        print the status (size) of nodes and elements
        """
        text_2D_3D = "3D" if self.is_3D else "2D"
        print(f"{text_2D_3D} mesh status:")
        print(f"{self.n_nodes} nodes, {self.n_elems} elements")

    def _check_element(self, element: np.ndarray) -> np.ndarray:
        """
        Check nodes element
        return nodes [x,y,z]

        Parameters
        ----------
        node : np.ndarray, optional
            nodes [x,y] ; shape (n_elem,3)
            nodes [x,y,z] ; shape (n_nodes,4)

        Returns
        -------
        np.ndarray
            nodes [x,y,z] ; shape (n_nodes,3)

        Raises
        ------
        TypeError
            raised if perm is not ndarray and of shape (n_tri,)
        """
        if not isinstance(element, np.ndarray):
            raise TypeError(f"Wrong type of {element=}, expected an ndarray")
        if element.ndim != 2:
            raise TypeError(
                f"Wrong shape of {element.shape=}, expected an ndarray with 2 dimensions"
            )
        if element.shape[1] not in [3, 4]:
            raise TypeError(
                f"Wrong shape of {element.shape=}, expected an ndarray of shape (n_nodes,2) or (n_nodes,3)"
            )

        return element

    def _check_node(self, node: np.ndarray) -> np.ndarray:
        """
        Check nodes shape
        return nodes [x,y,z]

        Parameters
        ----------
        node : np.ndarray, optional
            nodes [x,y] ; shape (n_nodes,2) (in that case z will be set 0)
            nodes [x,y,z] ; shape (n_nodes,3)

        Returns
        -------
        np.ndarray
            nodes [x,y,z] ; shape (n_nodes,3)

        Raises
        ------
        TypeError
            raised if perm is not ndarray and of shape (n_tri,)
        """
        if not isinstance(node, np.ndarray):
            raise TypeError(f"Wrong type of {node=}, expected an ndarray")
        if node.ndim != 2:
            raise TypeError(
                f"Wrong shape of {node.shape=}, expected an ndarray with 2 dimensions"
            )
        if node.shape[1] not in [2, 3]:
            raise TypeError(
                f"Wrong shape of {node.shape=}, expected an ndarray of shape (n_nodes,2) or (n_nodes,3)"
            )
        # convert nodes [x,y] to nodes [x,y,0]
        if node.shape[1] == 2:
            node = np.hstack((node, np.zeros((node.shape[0], 1))))

        return node

    def get_valid_perm_array(
        self, perm: Union[int, float, complex, np.ndarray] = 1.0
    ) -> np.ndarray:
        """
        Return a permittivity NDArray on element

        Parameters
        ----------
        perm : Union[int, float, complex, np.ndarray], optional
            Permittivity on elements ; shape (n_elems,), by default `None`.
            If `None`, a uniform permittivity on elements with a value 1 will be used.
            If perm is int or float, uniform permittivity on elements will be used.

        Returns
        -------
        np.ndarray
            permittivity on elements ; shape (n_elems,)

        Raises
        ------
        np.ndarray
            ndarray has a shape (n_elems,)
        """

        if perm is None:
            return np.ones(self.n_elems, dtype=float)
        elif isinstance(perm, (int, float)):
            return np.ones(self.n_elems, dtype=float) * perm
        elif isinstance(perm, complex):
            return np.ones(self.n_elems, dtype=complex) * perm

        if not isinstance(perm, np.ndarray) or perm.shape != (self.n_elems,):
            raise TypeError(
                f"Wrong type/shape of {perm=}, expected an ndarray; shape ({self.n_elems}, )"
            )
        return perm

    def _check_ref_node(self, ref: int = 0) -> int:
        """
        Return a valid reference electrode node

        Parameters
        ----------
        ref : int, optional
            node number of reference node, by default 0
            If the choosen node is on electrode node, a node-list in
            np.arange(0, len(el_pos)+1) will be checked iteratively until
            a non-electrode node is selected.

        returns
        -------
        int
            valid reference electrode node
        """
        default_ref = np.setdiff1d(np.arange(len(self.el_pos) + 1), self.el_pos)[0]
        return ref if ref not in self.el_pos else int(default_ref)
        # assert ref < self.n_nodes

    def set_ref_node(self, ref: int = 0) -> None:
        """
        Set reference electrode node

        Parameters
        ----------
        ref : int, optional
            node number of reference electrode
        """
        self.ref_node = self._check_ref_node(ref)

    @property
    def perm_array(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            ndarray has a shape (n_elems,)
        """
        return self.get_valid_perm_array(self.perm)

    @property
    def n_nodes(self) -> int:
        """
        Returns
        -------
        int
            number of nodes contained in the mesh
        """
        return self.node.shape[0]

    @property
    def n_elems(self) -> int:
        """
        Returns
        -------
        int
            number of elements contained in the mesh
        """
        return self.element.shape[0]

    @property
    def n_vertices(self) -> int:
        """
        Returns
        -------
        int
            number of vertices of the elements contained in the mesh
        """
        return self.element.shape[1]

    @property
    def n_el(self) -> int:
        """
        Returns
        -------
        int
            number of electrodes
        """
        return self.el_pos.shape[0]

    @property
    def elem_centers(self):
        """
        Returns
        -------
        np.ndarray
            center of the nodes [x,y,z]; shape (n_elems,3)
        """
        return np.mean(self.node[self.element], axis=1)

    @property
    def dtype(self):
        """
        Returns
        -------
        Type
            data type of permmitivity
        """
        if isinstance(self.perm, (int, float)):
            return float
        elif isinstance(self.perm, complex):
            return complex
        elif isinstance(self.perm, np.ndarray):
            return self.perm.dtype

    @property
    def is_3D(self) -> bool:
        """
        Returns
        -------
        np.ndarray
            True if the mesh is a 3D mesh (use elements with 4 vertices)
        """
        return self.n_vertices == 4

    @property
    def is_2D(self) -> bool:
        """
        Returns
        -------
        np.ndarray
            True if the mesh is a 2D mesh (use elements with 3 vertices)
        """
        return self.n_vertices == 3


def create(
    n_el: int = 16,
    fd: Callable = shape.circle,
    fh: Callable = shape.area_uniform,
    h0: float = 0.1,
    p_fix: Optional[np.ndarray] = None,
    bbox: Optional[np.ndarray] = None,
) -> PyEITMesh:
    """
    Generating 2D/3D meshes using distmesh (pyEIT built-in)

    Parameters
    ----------
    n_el: int
        number of electrodes (point-type electrode)
    fd: function
        distance function (circle in 2D, ball in 3D)
    fh: function
        mesh size quality control function
    p_fix: NDArray
        fixed points
    bbox: NDArray
        bounding box
    h0: float
        initial mesh size, default=0.1

    Returns
    -------
    PyEITMesh
        mesh object
    """

    # test conditions if fd or/and bbox are none

    if bbox is None:
        if fd != shape.ball:
            bbox = np.array([[-1, -1], [1, 1]])
        else:
            bbox = np.array([[-1.2, -1.2, -1.2], [1.2, 1.2, 1.2]])

    # list is converted to Numpy array so we can use it then (calling shape method..)
    bbox = np.array(bbox)
    n_dim = bbox.shape[1]  # bring dimension

    if n_dim not in [2, 3]:
        raise TypeError("distmesh only supports 2D or 3D")
    if bbox.shape[0] != 2:
        raise TypeError("please specify lower and upper bound of bbox")

    if p_fix is None:
        if n_dim == 2:
            if fd == shape.thorax:
                p_fix = shape.thorax_pfix
            elif fd == shape.head_symm:
                p_fix = shape.head_symm_pfix
            elif fd == shape.lshape:
                p_fix = shape.lshape_pfix
                h0 = 0.15
            else:
                p_fix = shape.fix_points_fd(fd, n_el=n_el)
        elif n_dim == 3:
            p_fix = shape.fix_points_ball(n_el=n_el)

    # 1. build mesh
    p, t = build(fd, fh, pfix=p_fix, bbox=bbox, h0=h0)
    # 2. check whether t is counter-clock-wise, otherwise reshape it
    t = check_order(p, t)
    # 3. generate electrodes, the same as p_fix (top n_el)
    el_pos = np.arange(n_el)
    return PyEITMesh(element=t, node=p, el_pos=el_pos, ref_node=0)


@dataclass
class PyEITAnomaly(ABC):
    """
    Pyeit Anomaly for simulation purpose
    """

    center: Union[np.ndarray, list]  # center of the anomaly
    perm: float = 1.0  # permittivity of the anomaly

    def __post_init__(self):
        if isinstance(self.center, list):
            self.center = np.array(self.center)

    @abstractmethod
    def mask(self, pts: np.ndarray) -> np.ndarray:
        """
        Return mask corresponding to the pts contained in the Anomaly
        """


@dataclass
class PyEITAnomaly_Circle(PyEITAnomaly):
    """
    Pyeit Anomaly for simulation purpose, 2D circle

    """

    r: float = 1.0  # radius of the circle

    def mask(self, pts: np.ndarray) -> Any:
        pts = pts[:, :2].reshape((-1, 2))
        return circle(pts, self.center[:2], self.r) < 0


@dataclass
class PyEITAnomaly_Ball(PyEITAnomaly):
    """
    Pyeit Anomaly for simulation purpose, 3D ball
    """

    r: float = 1.0  # radius of the ball

    def mask(self, pts: np.ndarray) -> Any:
        pts = pts.reshape((-1, 3))
        return ball(pts, self.center, self.r) < 0


def set_perm(
    mesh: PyEITMesh,
    anomaly: Union[PyEITAnomaly, List[PyEITAnomaly]],
    background: Optional[float] = None,
) -> PyEITMesh:
    """wrapper for pyEIT interface

    Note
    ----
    update permittivity of mesh, if specified.

    Parameters
    ----------
    mesh: PyEITMesh
        mesh object
    anomaly: Union[PyEITAnomaly, List[PyEITAnomaly]], optional
        anomaly object or list of anomalyobject contains,
        all permittivity on triangles whose distance to (x,y) are less than (d)
        will be replaced with a new value, 'perm' may be a complex value.
    background: float, optional
        set background permittivity

    Returns
    -------
    PyEITMesh
        mesh object
    """
    if isinstance(anomaly, PyEITAnomaly):
        anomaly = [anomaly]
    if isinstance(mesh.perm, np.ndarray):
        perm = mesh.perm.copy()
    else:
        perm = mesh.perm * np.ones(mesh.n_elems)
    # reset background if needed
    if background is not None:
        perm = background * np.ones(mesh.n_elems)

    # complex-valued permmitivity
    for an in anomaly:
        if np.iscomplex(an.perm):
            perm = perm.astype("complex")
            break

    # assign anomaly values (for elements in regions)
    tri_centers = mesh.elem_centers
    for an in anomaly:
        mask = an.mask(tri_centers)
        perm[mask] = an.perm

    return PyEITMesh(
        node=mesh.node,
        element=mesh.element,
        perm=perm,
        el_pos=mesh.el_pos,
        ref_node=mesh.ref_node,
    )


def layer_circle(n_el: int = 16, n_fan: int = 8, n_layer: int = 8) -> PyEITMesh:
    """generate mesh on unit-circle"""
    model = MeshCircle(n_fan=n_fan, n_layer=n_layer, n_el=n_el)
    pts, tri, el_pos = model.create()
    # perm = np.ones(tri.shape[0]) not need anymore as handled in PyEITMesh
    return PyEITMesh(element=tri, node=pts, el_pos=el_pos)
