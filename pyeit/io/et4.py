# pylint: disable=no-member, invalid-name, unused-argument
# pylint: disable=duplicate-code
"""
Load .et4 file into mem (experimental).
This file structure may be modified in near future.

liubenyuan@gmail.com
2015-07-23, 2017-09-26
"""

# stdlib
import struct

# numerical
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def to_df(fstr, ex_mtx=None, step=1, et3_compatible=False,
          output_resistor=False):
    """ convert data to df """
    pass


def to_csv(fstr_from, fstr_to, ex_mtx=None, step=1, et3_compatible=False,
           output_resistor=False):
    """ convert data to csv """
    data = load(fstr_from,
                ex_mtx=ex_mtx,
                step=step,
                et3_compatible=et3_compatible,
                output_resistor=output_resistor)

    fstr_re_to = fstr_to.replace('csv', 're.csv')
    df = pd.DataFrame(data.real, index=None, columns=None)
    df.to_csv(fstr_re_to, columns=None, header=False, index=False)

    fstr_im_to = fstr_to.replace('csv', 'im.csv')
    df = pd.DataFrame(data.imag, index=None, columns=None)
    df.to_csv(fstr_im_to, columns=None, header=False, index=False)


def load(fstr, ex_mtx=None, step=1, et3_compatible=False,
         output_resistor=False):
    # pylint: disable=too-many-locals
    """
    .et4 is my private file format for XEIT-ng system

    try to read data and parse FILE HEADER
        [-- 128uint parameters + (256 float RE) + (256 float IM) --]

    Returns
    -------
    NDArray
        return data (complex valued)

    Notes
    -----
    in .et4, a frame consists many (16) excitations, and in each excitation
    ADC samples are arranged as AD1, AD2, ..., AD16, consecuetively. so
    (AD1, AD2, ..., AD16) (AD1, AD2, ..., AD16) ... (AD1, AD2, ..., AD16)

    "see hardware section in deit.pdf"

    when the excitation electrodes are marked as A, B,
    in order to be compatible with .et3, you should
    1. automatically zero-out 4 measures where A, A-1, B, B-1
    2. rearrange all the measures so that in each excitation,
       the first measurement always start from (include) 'A'
    3. divide 'diff Voltage' / Current = Resistor (optional)
    """
    # 1. get .et4 file length
    nbytes = et4_tell(fstr)

    # 2. get nframes (a frame = (128 + 256 + 256) 640)
    frame_size = 640*4
    header_size = 128*4
    nframe = int((nbytes) / (frame_size))

    # 1. prepare storage
    x = np.zeros((nframe, 512), dtype=np.float)

    # 3. unpack data and extract parameters
    with open(fstr, 'rb') as fh:
        for i in range(nframe):
            d = fh.read(frame_size)
            x[i] = np.array(struct.unpack('512f', d[header_size:]))

    data = x[:, :256] + 1j*x[:, 256:]

    # electrode re-arranged the same as .et3 file
    if et3_compatible:
        vindex, cindex = zero_rearrange_index(ex_mtx)
        dout = data[:, vindex]
        # R = diff_V / I
        if output_resistor:
            numLines = len(cindex)
            M = int(len(vindex) / numLines)
            for i in range(numLines):
                current = data[:, cindex[i]]
                for j in range(M):
                    voltage = i*M + j
                    dout[:, voltage] = dout[:, voltage] / current
    else:
        dout = data

    return dout


def load_info(fstr):
    """load info headers from XEIT"""
    # 1. get .et4 file length
    et4_len = et4_tell(fstr)

    # 2. get nframes (a frame = (128 + 256 + 256) 640)
    frame_size = 640*4
    nframe = int((et4_len) / (frame_size))

    # 1. prepare storage
    info = np.zeros((nframe, 128))

    # 3. unpack data and extract parameters
    with open(fstr, 'rb') as fh:
        for i in range(nframe):
            d = fh.read(frame_size)
            info[i, :] = np.array(struct.unpack('33if94i', d[:512]))

    return info


def load_raw(fstr, ex_mtx=None, et3_compatible=False,
             output_resistor=False):
    """ return raw data (complex valued) [256 x Nsamples] """
    pass


def et4_tell(fstr):
    """ check the filetype of et4 """
    with open(fstr, 'rb') as fh:
        fh.seek(0, 2)  # move the cursor to the end (2) of the file
        file_len = fh.tell()

    return file_len


def zero_rearrange_index(ex_mtx):
    """
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

    v, c = [], []  # non-zero diff-pairs and current values
    for k in range(num_lines):
        if ab_scan:
            ex_pat = ex_mtx[k, :].ravel()
            a = np.where(ex_pat == 1)[0][0]
            b = np.where(ex_pat == -1)[0][0]
        else:
            a = k                       # positive excitation
            b = (a + el_dist) % num_el  # negative excitation
        ap = (a - 1) % num_el  # positive adjacent
        bp = (b - 1) % num_el  # negative adjacent
        # print(A, B, Ap, Bp)
        c.append(k*num_el + b)
        for i in range(num_el):
            # re-order data start after A
            j = (i + a) % num_el
            if not(j == a or j == b or j == ap or j == bp):
                v.append(k*num_el + j)

    return v, c


if __name__ == "__main__":

    # .et4 file
    et_file = "../../data/s00-02.et4"

    # load data
    et4_data = load(fstr=et_file,
                    et3_compatible=True,
                    output_resistor=False)
    print(et4_data.shape)

    ti_real = np.real(et4_data).sum(axis=1)/192.0
    ti_imag = np.imag(et4_data).sum(axis=1)/192.0
    ti = np.sqrt(ti_real**2 + ti_imag**2)
    print("max = ", np.max(ti))
    print("min = ", np.min(ti))

    xlim = 1000
    if ti.shape[0] < 1000:
        xlim = ti.shape[0]

    # plot
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(211)
    ax.plot(ti_real, 'b-')
    axt = ax.twinx()
    axt.plot(ti_imag, 'r-')
    ax.set_xlim([0, xlim])
    ax.grid('on')

    ax2 = fig.add_subplot(212)
    ax2.plot(ti, 'r-')
    ax2.grid('on')
    ax2.set_xlim([0, xlim])
    plt.show()
    # fig.savefig('s00-03-80k.png')
