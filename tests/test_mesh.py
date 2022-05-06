# test for mesh.py
import numpy as np
import pyeit.mesh as pm


def test_shape_circle():
    """unit circle mesh"""

    def _fd(pts):
        """shape function"""
        return pm.shape.circle(pts, pc=[0, 0], r=1.0)

    def _fh(pts):
        """distance function"""
        r2 = np.sum(pts**2, axis=1)
        return 0.2 * (2.0 - r2)

    # build fix points, may be used as the position for electrodes
    num = 16
    p_fix = pm.shape.fix_points_circle(ppl=num)
    p, t = pm.distmesh.build(_fd, _fh, pfix=p_fix, h0=0.12)
    assert p.shape[0] > 0
    assert t.shape[0] > 0
    assert pm.check_ccw(p, t)


def test_thorax():
    """Thorax mesh"""
    p, t = pm.distmesh.build(
        fd=pm.shape.thorax,
        fh=pm.shape.area_uniform,
        pfix=pm.shape.thorax_pfix,
        h0=0.15,
    )
    assert p.shape[0] > 0
    assert t.shape[0] > 0
    assert pm.check_ccw(p, t)


def test_head_symm():
    """Head mesh"""
    p, t = pm.distmesh.build(
        fd=pm.shape.head_symm,
        fh=pm.shape.area_uniform,
        pfix=pm.shape.head_symm_pfix,
        h0=0.15,
    )
    assert p.shape[0] > 0
    assert t.shape[0] > 0
    assert pm.check_ccw(p, t)


if __name__ == "__main__":
    pass
