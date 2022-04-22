import unittest
import pyeit.eit.fem


class TestVoltageMeterMethod(unittest.TestCase):
    def test_std(self):
        el = 16
        step = 1
        ex_mat = pyeit.eit.fem.eit_scan_lines(ne=el, dist=step)
        parser = None
        res = [
            pyeit.eit.fem.voltage_meter(ex_line=line, n_el=el, step=step, parser=parser)
            for line in ex_mat
        ]
        # test size of each array > 0
        self.assertTrue(all(r.size > 0 for r in res))

    def test_meas_current(self):
        el = 16
        step = 1
        ex_mat = pyeit.eit.fem.eit_scan_lines(ne=el, dist=step)
        parser = "meas_current"
        res = [
            pyeit.eit.fem.voltage_meter(ex_line=line, n_el=el, step=step, parser=parser)
            for line in ex_mat
        ]
        # test size of each array > 0
        self.assertTrue(all(r.size > 0 for r in res))


class TestVoltageMeterNdMethod(unittest.TestCase):
    def test_std(self):
        el = 16
        step = 1
        ex_mat = pyeit.eit.fem.eit_scan_lines(ne=el, dist=step)
        parser = None
        res = pyeit.eit.fem.voltage_meter_nd(
            ex_mat=ex_mat, n_el=el, step=step, parser=parser
        )
        # test size of each array > 0
        self.assertTrue(all(r.size > 0 for r in res))

    def test_meas_current(self):
        el = 16
        step = 1
        ex_mat = pyeit.eit.fem.eit_scan_lines(ne=el, dist=step)
        parser = "meas_current"
        res = pyeit.eit.fem.voltage_meter_nd(
            ex_mat=ex_mat, n_el=el, step=step, parser=parser
        )
        # test size of each array > 0
        self.assertTrue(all(r.size > 0 for r in res))


if __name__ == "__main__":
    unittest.main()
