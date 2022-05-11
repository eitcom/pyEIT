# test for eit
import unittest
import numpy as np

from pyeit.eit.fem import EITForward
import pyeit.mesh as mesh
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
import pyeit.eit.protocol as protocol
import pyeit.eit


def eit_loc_eval(ds: np.ndarray, mesh_obj: mesh.PyEITMesh, mode: str = "element"):
    """
    mode: string, default "element"
        if mode = "element", ds are values on element
        if mode = "node", ds are values on node (BP)
    """
    loc = np.argmax(np.abs(ds))
    if mode == "node":
        loc_xyz = mesh_obj.node[loc]
    else:
        loc_xyz = mesh_obj.elem_centers[loc]

    ds_max = ds[loc]
    ds_sign = np.sign(ds_max)

    return loc_xyz, ds_max, ds_sign


class TestFem(unittest.TestCase):
    def setUp(self):
        n_el = 16
        self.n_el = n_el
        self.mesh_obj = mesh.create(self.n_el, h0=0.1)

        # set anomaly
        self.anomaly = {"center": [0.5, 0.5], "r": 0.1, "perm": 10.0, "sign": True}
        anomaly = PyEITAnomaly_Circle(
            center=self.anomaly["center"],
            r=self.anomaly["r"],
            perm=self.anomaly["perm"],
        )
        self.mesh_new = mesh.set_perm(self.mesh_obj, anomaly=anomaly, background=1.0)
        self.protocol_obj = protocol.create(
            n_el, dist_exc=1, step_meas=1, parser_meas="std"
        )

        # calculate simulated data
        self.fwd = EITForward(self.mesh_obj, self.protocol_obj)
        self.v0 = self.fwd.solve_eit()
        self.v1 = self.fwd.solve_eit(perm=self.mesh_new.perm)

    def test_bp(self):
        """test back projection"""
        eit = pyeit.eit.bp.BP(self.mesh_obj, self.protocol_obj)
        # setup BP with no prior
        eit.setup(weight="none", perm=1)
        ds = 192.0 * eit.solve(self.v1, self.v0, normalize=False)

        # evaluate
        loc, ds_max, ds_sign = eit_loc_eval(ds, self.mesh_obj, mode="node")
        # print(loc, ds_max, ds_sign)
        loc = loc[: len(self.anomaly["center"])]
        dist = np.linalg.norm(loc - self.anomaly["center"])

        self.assertTrue(ds_sign == int(self.anomaly["sign"]))
        self.assertTrue(dist < self.anomaly["r"])

    def test_jac(self):
        """test jac"""
        eit = pyeit.eit.jac.JAC(self.mesh_obj, self.protocol_obj)
        eit.setup(p=0.5, lamb=0.01, method="kotre", perm=1, jac_normalized=True)
        ds = eit.solve(self.v1, self.v0, normalize=True)

        # evaluate
        loc, ds_max, ds_sign = eit_loc_eval(ds, self.mesh_obj, mode="element")
        # print(loc, ds_max, ds_sign)
        loc = loc[: len(self.anomaly["center"])]
        dist = np.linalg.norm(loc - self.anomaly["center"])

        self.assertTrue(ds_sign == int(self.anomaly["sign"]))
        self.assertTrue(dist < self.anomaly["r"])

    def test_svd(self):
        """test SVD"""
        eit = pyeit.eit.svd.SVD(self.mesh_obj, self.protocol_obj)
        eit.setup(n=50, method="svd", perm=1, jac_normalized=True)
        ds = eit.solve(self.v1, self.v0, normalize=True)
        # evaluate
        loc, ds_max, ds_sign = eit_loc_eval(ds, self.mesh_obj, mode="element")
        # print(loc, ds_max, ds_sign)
        loc = loc[: len(self.anomaly["center"])]
        dist = np.linalg.norm(loc - self.anomaly["center"])

        self.assertTrue(ds_sign == int(self.anomaly["sign"]))
        # more tolerance on SVD
        self.assertTrue(dist < 2 * self.anomaly["r"])

    def test_greit(self):
        """test GREIT"""
        eit = pyeit.eit.greit.GREIT(self.mesh_obj, self.protocol_obj)
        eit.setup(p=0.50, lamb=0.01, perm=1, jac_normalized=True)
        ds = eit.solve(self.v1, self.v0, normalize=True)
        x, y, ds = eit.mask_value(ds, mask_value=np.NAN)

        # evaluate GREIT
        loc = np.where(np.abs(ds) == np.nanmax(np.abs(ds)))
        center = np.array([x[loc][0], y[loc][0]])
        ds_sign = np.sign(ds[loc][0])
        # print(loc, center, ds_sign)
        dist = np.linalg.norm(center - self.anomaly["center"])

        self.assertTrue(ds_sign == int(self.anomaly["sign"]))
        self.assertTrue(dist < self.anomaly["r"])


if __name__ == "__main__":
    unittest.main()
