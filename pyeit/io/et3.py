# pylint: disable=no-member, invalid-name, too-many-instance-attributes
"""
load .et3, .et0 file into mem
using pack and unpack together with regular expression to filter out data.
"""
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from os.path import splitext
import re
from struct import unpack

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates
# pandas-0.21.0:
# from pandas.tseries import converter
# converter.register()


class ET3(object):
    """ et0 and et3 file parser"""

    def __init__(self, file_name, et_type='auto', trim=True, verbose=False):
        """
        initialize file parser (supports .et3, .et0)
        read data and parse FILE HEADER.

        Parameters
        ----------
        file_name : basestring
            file path
        et_type : string
            file types, 'et0' or 'et3', default is 'et3'
        trim : bool
            if True, trim EIT data from 256 down to
            206 (adjacent stimulation) or 192 (opposition stimulation)
        verbose : bool
            print debug messages
        """
        # choose file type (auto infer extension)
        if et_type not in ['et0', 'et3']:
            et_type = splitext(file_name)[1][1:]
        if verbose:
            print('file type is treated as %s' % et_type)
        # tell et0/et3 file information
        self.params = et_tell(file_name, et_type)
        self.file_name = file_name
        self.et_type = et_type
        self.trim = trim
        self.verbose = verbose

        self.offset = self.params['offset']
        self.nframe = self.params['nframe']
        # make parameter values valid
        if self.params['current'] > 1250 or self.params['current'] <= 0:
            if verbose:
                print('ET: current (%d) out of range', self.params['current'])
            # default current = 750 uA
            self.params['current'] = 750
        if self.params['gain'] not in [0, 1, 2, 3, 4, 5, 6, 7]:
            if verbose:
                print('ET: gain (%d) out of range', self.params['gain'])
            # default gain control = 3
            self.params['gain'] = 3

        # print debug information
        if verbose:
            for k in self.params.keys():
                print('%s: %s' % (k, self.params[k]))

        # constant variables
        # each frame = header + 2x256 (Re, Im) doubles
        self.header_size = 1024
        self.data_num = (2 * 256)
        self.data_size = self.data_num * 8
        self.frame_size = self.header_size + self.data_size

        # load et3 files (RAW data are complex-valued)
        self.data = self.load()

    def load(self):
        """
        load RAW data

        old-type (Bytes) :
            -- 4096 FILE HEADER --
            -- 1024 frame-header + (256 double RE) + (256 double IM) --
        new-type (Bytes) :
            -- 1024 frame-header + (256 double RE) + (256 double IM) --

        Returns
        -------
        data : NDArray
            complex-valued ndarray
        """
        x = np.zeros((self.nframe, self.data_num), dtype=np.double)

        # convert
        with open(self.file_name, 'rb') as fh:
            # skip offset
            fh.read(self.offset)

            # read data frame by frame
            for i in range(self.nframe):

                # get frame data
                d = fh.read(self.frame_size)

                # parse data every frame and store in a row of x
                x[i] = np.array(unpack('512d', d[self.header_size:]))

        # convert Re, Im to complex numbers
        raw_data = x[:, :256] + 1j * x[:, 256:]

        # [PS] remove all zeros columns in raw_data if trim is True
        if self.trim:
            idx = trim_pattern()
            data = raw_data[:, idx]
        else:
            data = raw_data

        # [PS] convert voltage to resistance (Ohms) if file is 'et0'
        # for 'et3', file is already in Ohms
        if self.et_type == 'et0':
            # byliu: current for et0 is locked to 750 uA
            # 1250/750 = 1.667
            data = data / self.params['current'] * 1.667

        return data

    def load_time(self, rel_date=None):
        """
        load timestamp from et file

        rel_date : relative date time, i.e., 1970/1/1 10:0:0
            if rel_date is provided, we do NOT read day from ET file.

        Notes
        -----
        Many files ending with .et0 are actual .et3 files, be warned.
        """
        # if user specify the date, use it!
        if rel_date is not None:
            # frame rate = 1 fps
            ta = np.arange(self.nframe)
        else:
            if self.et_type == 'et0':
                rel_date = '1994/1/1'
                # frame rate = 1 fps
                ta = np.arange(self.nframe)
            elif self.et_type == 'et3':
                # December 30, 1899 is the base date. (EXCEL format)
                rel_date = '1899/12/30'
                ta = np.zeros(self.nframe, dtype='double')
                # read days from frame header
                with open(self.file_name, 'rb') as fh:
                    fh.read(self.offset)
                    for i in range(self.nframe):
                        # read frame data
                        d = fh.read(self.frame_size)
                        t = et3_date(d)
                        ta[i] = t
                # convert days to seconds
                ta = ta * 86400.0

        # convert to pandas datetime
        ts = pd.to_datetime(rel_date) + pd.to_timedelta(ta, 's')

        return ts

    def reload(self):
        """reload data using different options"""
        pass

    def to_df(self, resample=None, rel_date=None):
        """convert raw data to pandas.DataFrame"""
        ts = self.load_time(rel_date=rel_date)
        df = pd.DataFrame(self.data, index=ts)
        # resample
        if resample is not None:
            df = df.resample(resample).mean()

        return df

    def to_csv(self):
        """
        save data in a .csv file (NotImplemented)

        this function is embedded in to pandas.DataFrame, simply call
        $ df.to_csv(file_to, columns=None, header=False, index=False)
        """
        pass


def et0_date():
    """
    extract date from .et0 file.
    for .et0 file date is stored in string (bytes) format (utf-16, CJK)
    i.e., '2016Y01M01D 11H08M30S', offset=4, length=42 (21*2) Bytes
    """
    raise NotImplementedError


def et3_date(d, verbose=False):
    """
    extract date from .et3 file.
    for .et3, date is stored at

    nVersion : int (4 Bytes)
    frame index : int (4 Bytes)
    time : double (8 bytes)

    returns
    -------
    time in days relative to julian date
    """
    if verbose:
        ftype = unpack('I', d[:4])
        print('file type: %d' % ftype)

    t = unpack('d', d[8:16])[0]
    return t


def et_tell(file_name, et_type='et3'):
    """
    Infer et0 or et3 file-type and header information

    Note: since 2016, all version are without a standalone file header.
    This function may be deprecated in near future.
    """
    if et_type == 'et3':
        _header_parser = et3_header
    else:
        _header_parser = et0_header

    with open(file_name, 'rb') as fh:
        # get file info (header)
        d = fh.read(1024)
        frequency, current, gain = _header_parser(d)

        # move the cursor to the end (2) of the file
        fh.seek(0, 2)
        et3_len = fh.tell()

    # extract file type (no longer needed since 2016)
    # 'deprecated' version has extra header (4096 Bytes)
    # 'new' : each frame is 5120 Bytes,
    # which has 1024 Bytes header + 256x2 Double data (4096 Bytes)
    is_header = (et3_len % 5120) != 0
    offset = 4096 if is_header else 0
    nframe = int((et3_len - offset) / 5120)

    # build output
    params = {'offset': offset,
              'nframe': nframe,
              'frequency': frequency,
              'current': current,
              'gain': gain}

    return params


def et0_header(d):
    """
    parse et0 header. Guess from binary dump (byliu)

    binary dump all (little endian, i.e., LSB 16 bit first):

    print('now dump')
    h_all = np.array(unpack('256I', d))
    for i in range(32):
        h_seg = h_all[i*8 + np.arange(8)]
        print(','.join('{:02x}'.format(x) for x in h_seg))
    """
    # unpack all
    header_offset = 48
    header_end = header_offset + 16
    h = np.array(unpack('8H', d[header_offset:header_end]))
    # print(','.join('{:02x}'.format(x) for x in h))

    # extract information
    frequency = np.int(h[1])
    current = np.int(h[3])
    gain = np.int(h[5])

    return frequency, current, gain


def et3_header(d):
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
    h = np.array(unpack('8I2f', d[header_offset:header_end]))

    # extract
    frequency = h[4]
    current = h[5]
    gain = h[6]

    return frequency, current, gain


def gain_table(gain):
    """
    Rescale data using if EIT is using programmable gain.
    """
    # Programmable Gain table (scale mapping), see MainFrm.cpp
    pg_table = {0: 4.112,
                1: 8.224,
                2: 16.448,
                3: 32.382,
                4: 64.764,
                5: 129.528,
                6: 257.514,
                7: 514}
    """
    new pg table (in dll) by Zhang Ge, 2017/12/06
    gain = {0: 0.08,
            1: 0.16,
            2: 0.32,
            3: 0.63,
            4: 1.26,
            5: 2.52,
            6: 5.01,
            7: 10.0}
    gain = 25.7 * keyvalue
    """

    # make sure gain is a valid key
    if gain not in pg_table.keys():
        scale = 1.
    else:
        # assume current = 1000 uA
        current = 1250.0
        scale = 2.5 * 1000000.0 / 32768.0 / current / pg_table[gain]

    return scale


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
        idx[j + 7] = False

    return idx


def get_date_from_folder(file_str):
    """
    get datetime from file folder of et3, i.e., 'DATA2015-01-29-16-57-30/'
    """
    f = file_str.strip()
    f = f[:-1]  # remove trailing '/'
    f = f.replace("DATA", "")
    # replace the 3rd occurrence of '-'
    w = [m.start() for m in re.finditer(r'-', f)][2]
    # before w do not change, after w, '-' -> ':'
    f = f[:w] + ' ' + f[w+1:].replace('-', ':')
    # now f becomes '2015-01-29 16:57:30'
    return pd.to_datetime(f)


def demo():
    """ demo shows how-to use et3 """
    file_name = '../../datasets/DATA.et3'
    # file_name = '../../datasets/RAWDATA.et0'
    # scale = 1000000.0 / (1250/750) = 600000.0

    # 1. using raw interface and calculate
    #    averaged transfer impedance from raw data
    # et3_data = load(fstr, verbose=True)
    # ts = load_time(fstr)
    # ati = np.abs(et3_data).sum(axis=1)/192.0

    # 2. using DataFrame interface, resample option:
    #    's' is seconds (default)
    #    'T' is minute
    et3 = ET3(file_name, verbose=True)
    df = et3.to_df(rel_date='2017/11/17')
    df['ati'] = np.abs(df).sum(axis=1) / 192.0

    # 3. plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # TODO: deprecate warning for pandas-0.21.0
    ax.plot(df.index.to_pydatetime(), df['ati'])
    ax.grid('on')

    # format time axis
    hfmt = dates.DateFormatter('%y/%m/%d %H:%M')
    ax.xaxis.set_major_formatter(hfmt)
    fig.autofmt_xdate()
    plt.show()

    return fig, ax


if __name__ == "__main__":
    demo()
