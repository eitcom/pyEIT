# pylint: disable=no-member, invalid-name
# pylint: disable=too-many-arguments, too-many-instance-attributes
"""
Load .et4 file into mem (experimental).
This file structure may be modified in near future.
"""
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from struct import unpack

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ET4:
    """.et4 file loader"""

    def __init__(
        self, file_name, ex_mtx=None, step=1, compatible=False, output_resistor=False
    ):
        """
        initialize .et4 handler.
        .et4 is an experimental file format for XEIT-ng system

        try to read data and parse FILE HEADER
            [-- 128 uint parameters + (256 float RE) + (256 float IM) --]

        Parameters
        ----------
        file_name : basestring
        ex_mtx : NDArray
            num_lines x 2 array
        step : int
            architecture of voltage meter
        compatible : bool
            if data output needs to be .et3 compatible
        output_resistor : bool
            convert voltage to current

        Returns
        -------
        NDArray
            return data (complex valued)

        Notes
        -----
        in .et4, a frame consists (16) excitations, and in each excitation
        ADC samples are arranged as AD1, AD2, ..., AD16, consecutively. so
        (AD1, AD2, ..., AD16) (AD1, AD2, ..., AD16) ... (AD1, AD2, ..., AD16)

        "see hardware section in eit.pdf, benyuan liu."

        when the excitation electrodes are marked as A, B,
        in order to be compatible with .et3, you should
        1. automatically zero-out 4 measures where A, A-1, B, B-1
        2. rearrange all the measures so that in each excitation,
           the first measurement always start from (include) 'A'
        3. divide 'diff Voltage' / Current = Resistor (optional)
        """
        self.file_name = file_name
        self.ex_mtx = ex_mtx
        self.step = step
        self.compatible = compatible
        self.output_resistor = output_resistor

        # 1. get .et4 file length
        nbytes = et4_tell(self.file_name)

        # 2. get nframes (a frame = (256 + 256d + 256d) = 5120 Bytes)
        self.info_num = 256
        self.data_num = 256 * 2
        self.header_size = self.info_num * 4
        self.frame_size = self.header_size + self.data_num * 8
        self.nframe = int((nbytes) / (self.frame_size))

        # 3. load data
        self.data = self.load()

    def load(self):
        """load RAW data"""
        # 1. prepare storage
        x = np.zeros((self.nframe, self.data_num), dtype=np.double)

        # 3. unpack data and extract parameters
        with open(self.file_name, "rb") as fh:
            for i in range(self.nframe):
                d = fh.read(self.frame_size)
                x[i] = np.array(unpack("512d", d[self.header_size :]))

        data = x[:, :256] + 1j * x[:, 256:]
        # electrode re-arranged the same as .et3 file
        if self.compatible:
            v_index, c_index = zero_rearrange_index(self.ex_mtx)
            dout = data[:, v_index]
            if self.output_resistor:
                # number of diff_V per stimulation
                M = int(len(v_index) / len(c_index))
                # current index is the same in each M measurements
                cm_index = np.repeat(c_index, M)
                # R = diff_V / I
                dout = dout / data[:, cm_index]
        else:
            dout = data

        return dout

    def load_info(self):
        """load info headers from xEIT"""
        # 1. prepare storage
        info = np.zeros((self.nframe, 256))

        # 3. unpack data and extract parameters
        with open(self.file_name, "rb") as fh:
            for i in range(self.nframe):
                d = fh.read(self.frame_size)
                info[i, :] = np.array(unpack("33if222i", d[: self.header_size]))

        return info

    def to_df(self, resample=None, rel_date=None, fps=20):
        """convert raw data to pandas.DataFrame"""
        if rel_date is None:
            rel_date = "2019/01/01"
        ta = np.arange(self.nframe) * 1.0 / fps
        ts = pd.to_datetime(rel_date) + pd.to_timedelta(ta, "s")
        df = pd.DataFrame(self.data, index=ts)
        # resample
        if resample is not None:
            df = df.resample(resample).mean()

        return df

    def to_csv(self):
        """save file to csv"""
        raise NotImplementedError()


def et4_tell(fstr):
    """check the filetype of et4"""
    with open(fstr, "rb") as fh:
        fh.seek(0, 2)  # move the cursor to the end (2) of the file
        file_len = fh.tell()

    return file_len


def zero_rearrange_index(ex_mtx):
    """
    (default mode: opposition stimulation)
    0. excitation electrodes are denoted by 'A' and 'B'
    1. for each excitation, REARRANGE all the data start from 'A'
    2. zero all the channels of A, A-1, B, B-1

    returns : re-ordered non-zero index, current index
    """
    if ex_mtx is None:
        num_lines, num_el, el_dist = 16, 16, 8
        ab_scan = False
    else:
        num_lines, num_el = ex_mtx.shape
        ab_scan = True

    v_index, c_index = [], []  # non-zero diff-pairs and current values
    for k in range(num_lines):
        if ab_scan:
            ex_pat = ex_mtx[k, :].ravel()
            a = np.where(ex_pat == 1)[0][0]
            b = np.where(ex_pat == -1)[0][0]
        else:
            a = k  # positive excitation
            b = (a + el_dist) % num_el  # negative excitation
        ap = (a - 1) % num_el  # positive adjacent
        bp = (b - 1) % num_el  # negative adjacent
        # print(A, B, Ap, Bp)
        c_index.append(k * num_el + b)
        for i in range(num_el):
            # re-order data start after A
            j = (i + a) % num_el
            if j not in (a, b, ap, bp):
                v_index.append(k * num_el + j)

    return v_index, c_index


if __name__ == "__main__":
    # .et4 file
    et_file = r"C:\xeit\eit_20190929-103342.et4"
    # et_file = "../../datasets/s00-02.et4"

    # load data
    et4 = ET4(et_file, compatible=True, output_resistor=False)
    et4_data = et4.data
    print(et4_data.shape)

    ti = et4_data.sum(axis=1) / 192.0
    ti_real = np.real(ti)
    ti_imag = np.imag(ti)
    ti_abs = np.sqrt(ti_real**2 + ti_imag**2)
    print("max = ", np.max(ti_abs))
    print("min = ", np.min(ti_abs))

    xlim = 1000
    if ti_abs.shape[0] < 1000:
        xlim = ti_abs.shape[0]

    # plot
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(211)
    ax.plot(ti_real, "b-")
    axt = ax.twinx()
    axt.plot(ti_imag, "r-")
    ax.set_xlim([0, xlim])
    ax.grid(True)

    ax2 = fig.add_subplot(212)
    ax2.plot(ti_abs, "r-")
    ax2.grid(True)
    ax2.set_xlim([0, xlim])
    plt.show()
    # fig.savefig('s00-03-80k.png')
