# pylint: disable=no-member, invalid-name, duplicate-code
"""
load .et3, .et0 file into mem
liubenyuan@gmail.com
2015-07-23, 2017-09-25
"""

# using pack and unpack together with regular expression to filter out data
import struct

# numerical
import numpy as np
import pandas as pd

# for visualization
import matplotlib.pyplot as plt
from matplotlib import dates


# load et3 into data frame
def to_df(file, trim=True, verbose=False, resample=None):
    """
    convert et3 file into pandas DataFrame
    """
    d = load(file, trim=trim, verbose=verbose)
    ts = load_time(file, verbose=verbose)

    df = pd.DataFrame(d, index=ts)
    if resample is not None:
        df = df.resample(resample).mean()

    return df


# .et3, .et0 file extension
#   return data (complex valued)
def load(fstr, trim=True, verbose=False):
    """
    try to read data and parse FILE HEADER

    current : float
        inject current (mA)

    old-type (Bytes) :
        -- 4096 FILE HEADER --
        -- 1024 frame-header + (256 double RE) + (256 double IM) --
    new-type (Bytes) :
        -- 1024 frame-header + (256 double RE) + (256 double IM) --
    """
    # get .et3 types
    nframe, _ = et3_tell(fstr, verbose=verbose)

    # each frame = header + 256x2 (Re, Im) doubles
    header_size = 1024
    data_size = (2*256)*4
    frame_size = header_size + 2*data_size
    x = np.zeros((nframe, 512), dtype=np.double)

    # convert
    with open(fstr, 'rb') as fh:
        for i in range(nframe):
            # get frame data
            d = fh.read(frame_size)

            # parse data every frame
            x[i, :] = np.array(struct.unpack('512d', d[header_size:]))

    # convert Re, Im to complex numbers
    raw_data = x[:, :256] + 1j*x[:, 256:]

    # if we need to remove all zeros columns in raw_data
    if trim:
        idx = trim_pattern()
        data = raw_data[:, idx]
    else:
        data = raw_data

    # Rescale data (do not needed)
    # scale = et3_gain(g)
    # data = data * scale

    return data


def et3_header(d, verbose=False):
    """
    Parse the header of et files, bytes order:

    nVersion : int (4 Bytes)
    frame index : int (4 Bytes)
    time : double (8 bytes)
    -- offset 3 double (24 Bytes) --

    Reconstruction Parameters
    -- offset 40 double (320 Bytes) --

    struct configinfo{
        DWORD dwDriveDelay;     // driver switch interval (us)
        DWORD dwMeasureDelay;   // measure switch interval (us)
        int nDrvNode;           // the distance of two driving electrodes
        int nMeaNode;           // the distance of two measure electrodes
        int nFrequency;         // Hertz
        int nCurrent;           // uA
        int nGain;              // Gain number
        int nElecNum;           // number of electrodes
        float fElecSize;        // size of electrodes
        float fPeriod;          // frame interval (s)
    };
    """
    header_offset = 360
    header_end = header_offset + 40
    h = np.array(struct.unpack('8I2f', d[header_offset:header_end]))

    # extract
    frequency = h[4]
    current = h[5]
    gain = h[6]

    return frequency, current, gain


def et3_gain(gain):
    """
    Rescale data using if EIT is using programmable gain.
    """
    # Gain table (scale mapping), see MainFrm.cpp
    gain_table = {0: 4.112,
                  1: 8.224,
                  2: 16.448,
                  3: 32.382,
                  4: 64.764,
                  5: 129.528,
                  6: 257.514,
                  7: 514}

    # make sure gain is a valid key
    if gain not in gain_table.keys():
        scale = 1.
    else:
        scale = 1000.0 * 2.5 / 32768.0 / gain_table[gain]

    return scale


def et3_tell(fstr, verbose=False):
    """
    Infer .et3 file-type

    Note: since 2016, all version are without file header.
    This function may be deprecated in near future.
    """
    with open(fstr, 'rb') as fh:
        # get file info (header)
        d = fh.read(1024)
        f, c, g = et3_header(d)
        if verbose:
            print('Freq=%d (Hz), Current=%d (uA), Gain=%d\n' % (f, c, g))

        # move the cursor to the end (2) of the file
        fh.seek(0, 2)
        et3_len = fh.tell()

    # extract file type (no longer needed since 2016)
    # 'deprecated' version has extra header (4096 Bytes)
    # 'new' : each frame is 1024 Bytes header + 256x2 Double (4096 Bytes)
    ftype = et3_len % (5*1024)
    offset = 4096 if ftype != 0 else 0
    nframe = int((et3_len-offset) / (5*1024))

    # build output
    para = {'frequency': f, 'current': c, 'gain': g}
    return nframe, para


def load_time(fstr, verbose=False):
    """ load timestamp from et file """
    nframe, _ = et3_tell(fstr)

    # each frame = header + 256x2 (Re, Im) doubles
    header_size = 1024
    data_size = (2*256)*4
    frame_size = header_size + 2*data_size

    # time-array storage
    ta = np.zeros(nframe, dtype=np.double)

    with open(fstr, 'rb') as fh:
        for i in range(nframe):
            # read frame data
            d = fh.read(frame_size)
            t = struct.unpack('d', d[8:16])[0]
            ta[i] = t

    # convert days to seconds (multiply number of seconds in a day)
    ta = np.round(ta * 86400)

    # December 30, 1899 is the base date.
    ts = pd.to_datetime('1899-12-30 00:00:00') + pd.to_timedelta(ta, 's')

    return ts


def trim_pattern():
    """
    Generate trim array (fixed) for .et3 and .et0,
    where idx is the indices of 0s

    0......0 0......0 0......0 etc.,
    """
    idx = np.ones(256, dtype=np.bool)
    for i in range(32):
        j = i * 8
        # exclude the stimulation indices (i.e., 0-8 stimulus)
        idx[j] = False
        idx[j+7] = False

    return idx


if __name__ == "__main__":
    """ demo shows how-to use et3 """
    fstr = '../../data/DATA.et3'

    # 1. using raw interface
    et3_data = load(fstr, verbose=True)
    ts = load_time(fstr, verbose=True)

    # averaged transfer impedance
    ati = np.abs(et3_data).sum(axis=1)/192.0

    # 2. using DataFrame interface, resample=
    #    's' is seconds (default)
    #    'T' is minute
    df = to_df(fstr)
    df['ati'] = np.abs(df).sum(axis=1)/192.0

    # 3. plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(df['ati'])
    ax.grid('on')

    # format time axis
    hfmt = dates.DateFormatter('%m/%d %H:%M')
    ax.xaxis.set_major_formatter(hfmt)
    fig.autofmt_xdate()
    plt.show()
