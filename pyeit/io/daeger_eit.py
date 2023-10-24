# pylint: disable=no-member, invalid-name, too-many-locals
"""
load daeger .eit files
reimplement daeger-eit in EIDORS3D (eidors_readdata.m)

You need to create a ex_mat using eit_scan_lines with 16 electrodes and
adjacent stimulation, rotate measurements in eit simulation.
"""
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
import os
import struct
import numpy as np


class DAEGER_EIT:
    """process daeger pulmovista .eit file"""

    def __init__(self, fname):
        """
        initialize

        Note
        ----
        Daeger using standard measuremants, basic_stim in EIDORS, where
        mk_stim_patterns(16, 1, [0, 1], [0, 1], ... \
                         {'no_meas_current','rotate_meas'}, 0.005);
        1, 16 electrodes, 1 ring,
        2, adjacent stimulation, adjacent measurement, [ex_mat, step=1]
        3, skip current carring electrodes,
        4, rotate measurements with stimulation patterns
        5, 1 unit current

        Parameters
        ----------
        file_name : basestring
            file path
        """
        self.fname = fname
        self.info = self.read_header(self.fname)
        self.ft = np.array([0.00098242, 0.00019607])  # estimated AA: 2016-04-07

    @staticmethod
    def read_header(fname, max_lines=50):
        """read information from header in text format"""
        fr = 0
        fmt = 0
        with open(fname, "r", encoding="ISO-8859-1") as fd:
            data = fd.readlines()
            lines = data[:max_lines]
            for line in lines:
                if "Framerate [Hz]" in line:
                    cell = line.split(":")
                    fr = int(cell[1])
                if "Format:" in line:
                    cell = line.split(":")
                    fmt = int(cell[1])
        # failed to find valid parameters
        if fr == 0:
            print("Frame rate could not be read, setting to 20")
            fr = 20
        if fmt == 0:
            print("Format could not be read, setting to 51")
            fmt = 51

        # find spc: bytes per frame
        daeger_spc = dict({31: 4112, 32: 3200, 51: 5495})
        if fmt not in daeger_spc.keys():
            print("Error, format version={} not supported".format(fmt))
        spc = daeger_spc[fmt]

        # get file length
        flen = os.path.getsize(fname)

        # get data offset
        with open(fname, "rb") as fh:
            b = fh.read(16)
            a = struct.unpack("8H", b)
            offset = a[2] + 16

        # number of frames
        nframe = int((flen - offset) / spc)

        # information in headers
        par = {
            "framerate": fr,
            "format": fmt,
            "spc": spc,
            "flen": flen,
            "offset": offset,
            "nframe": nframe,
        }

        return par

    def read_data(self):
        """read data frame by frame"""
        nframe = self.info["nframe"]
        data = np.zeros((nframe, 600), dtype=np.double)
        with open(self.fname, "rb") as fh:
            fh.seek(self.info["offset"])
            for i in range(nframe):
                d = fh.read(self.info["spc"])
                data[i] = struct.unpack("600d", d[:4800])

        return data

    def load(self):
        """convert data in to measurements (voltages)"""
        data = self.read_data()
        vv = self.ft[0] * data[:, :208] - self.ft[1] * data[:, 322:530]
        return vv

    def to_df(self):
        """
        Require a data structure document within a frame of daeger pulmovista,
        to extract the time stamp information and build the time series index.
        """
        raise NotImplementedError()


if __name__ == "__main__":
    file_name = r"./ID_SC_10_001.eit"
    model = DAEGER_EIT(fname=file_name)
    print(model.info)
    # test loading from daeger EIT
    model.load()
