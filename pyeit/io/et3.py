# pylint: disable=no-member, invalid-name, too-many-instance-attributes
"""
load .et3, .et0 file into mem
using pack and unpack together with regular expression to filter out data.

The .et3, .et0 file type was developed by FMMU EIT group.
Please cite the following paper if you are using et3 in your research:
    Fu, Feng, et al. "Use of electrical impedance tomography to monitor
    regional cerebral edema during clinical dehydration treatment."
    PloS one 9.12 (2014): e113202.
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
        self.file_name = file_name
        self.trim = trim
        self.verbose = verbose
        # choose file type (auto infer extension)
        if et_type not in ['et0', 'et1', 'et3']:
            et_type = splitext(file_name)[1][1:]

        # try read the file type by the information on extension, default: et0
        self.params = et_tell(file_name, et_type)

        # check if it is the right file-format
        current = self.params['current']
        if current > 1250 or current <= 0:
            if verbose:
                print('ET: file type mismatch')
            # force file type to ET3, re-parse the information
            et_type = 'et3'
            print('ET: current = %d is out of range (0, 1250]' % current)
            self.params = et_tell(file_name, et_type)

        self.et_type = et_type
        self.version = self.params['version']
        self.offset = self.params['offset']
        self.nframe = self.params['nframe']
        self.npar = 8  # number of maximum parameters [0622]

        # check if gain is correct
        gain = self.params['gain']
        if gain not in [0, 1, 2, 3, 4, 5, 6, 7]:
            print('ET: gain = %d is out of range, set to 3' % gain)
            # default gain control = 3
            self.params['gain'] = 3

        # print debug information
        if verbose:
            for k in self.params:
                print('%s: %s' % (k, self.params[k]))

        # constant variables
        # each frame = header + 2x256 (Re, Im) doubles
        self.header_size = 1024
        self.data_num = (2 * 256)
        self.data_size = self.data_num * 8
        self.frame_size = self.header_size + self.data_size

        # load et3 files (RAW data are complex-valued)
        self.data, self.dp = self.load()

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
        xp = np.zeros((self.nframe, self.npar), dtype=np.double)

        # convert
        with open(self.file_name, 'rb') as fh:
            # skip frame offset, if any (et0)
            fh.read(self.offset)

            # read data frame by frame
            for i in range(self.nframe):

                # get frame data
                d = fh.read(self.frame_size)

                # parse aux ADC
                dp = d[960:self.header_size]
                xp[i] = np.array(unpack('8d', dp))

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

        # [PS] convert ADC to resistance (Ohms) if file is 'et0'
        # for 'et3', file is already in Ohms
        if self.et_type == 'et0':
            scale = gain_table(self.params['gain'], self.params['current'])
            data = data * scale

        return data, xp

    def load_time(self, rel_date=None, fps=1):
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
            ta = np.arange(self.nframe) * 1.0 / fps
        else:
            if self.et_type == 'et0':
                rel_date = '1994/1/1'
                # frame rate = 1 fps
                ta = np.arange(self.nframe)
            elif self.et_type == 'et3':
                # December 30, 1899 is the base date. (EXCEL format)
                rel_date = '1899/12/30'
                # 'ta' should be int/long to keep time resolution to 's'
                ta = np.zeros(self.nframe, dtype='int64')
                # read days from a frame header
                with open(self.file_name, 'rb') as fh:
                    fh.read(self.offset)
                    for i in range(self.nframe):
                        # read frame data
                        d = fh.read(self.frame_size)
                        t = et3_date(d)
                        # convert days to seconds
                        ta[i] = t * 86400

        # convert to pandas datetime
        ts = pd.to_datetime(rel_date) + pd.to_timedelta(ta, 's')

        return ts

    def reload(self):
        """reload data using different options"""
        pass

    def to_df(self, resample=None, rel_date=None, fps=1):
        """convert raw data to pandas.DataFrame"""
        ts = self.load_time(rel_date=rel_date, fps=fps)
        df = pd.DataFrame(self.data, index=ts)

        # resample
        if resample is not None:
            df = df.resample(resample).mean()

        return df

    def to_dp(self, resample=None, rel_date=None, fps=1, filter=False):
        """convert raw parameters to pandas.DataFrame"""
        ts = self.load_time(rel_date=rel_date, fps=fps)
        columns = ['tleft', 'tright', 'nt_s', 'rt_s', 'r0', 'r1', 'r2', 'r3']
        dp = pd.DataFrame(self.dp, index=ts, columns=columns)

        if filter:
            # correct temperature (temperature cannot be 0)
            dp.loc[dp['tleft']==0, 'tleft'] = np.nan
            dp.loc[dp['tright']==0, 'tright'] = np.nan
            dp.loc[dp['nt_s']==0, 'nt_s'] = np.nan
            dp.loc[dp['rt_s']==0, 'rt_s'] = np.nan

            dp.tleft = med_outlier(dp.tleft)
            dp.tright = med_outlier(dp.tright)
            dp.nt_s = med_outlier(dp.nt_s)
            dp.rt_s = med_outlier(dp.rt_s)

        # resample
        if resample is not None:
            dp = dp.resample(resample).mean()

        return dp

    def to_csv(self):
        """
        save data in a .csv file (NotImplemented)

        this function is embedded in to pandas.DataFrame, simply call
        $ df.to_csv(file_to, columns=None, header=False, index=False)
        """
        pass


def med_outlier(d, window=17):
    med = d.rolling(window, center=False).median()
    std = d.rolling(window, center=False).std()
    std[std==np.nan] = 0.0
    # replace med with d for outlier removal
    df = med[(d <= med+3*std) & (d >= med-3*std)]
    return df


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
        _header_parser_func = et3_header
    else:
        _header_parser_func = et0_header

    with open(file_name, 'rb') as fh:
        # get file info (header)
        d = fh.read(1024)
        params = _header_parser_func(d)

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
    params['offset'] = offset
    params['nframe'] = nframe

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

    current can also be read out in each switches:

    header_offset = 84
    h = np.array(unpack('4H', d[header_offset:header_offset+8]))
    nGain, nCurrent = h[1], h[3]
    print(nGain, nCurrent)
    """
    # unpack all
    header_offset = 48
    header_end = header_offset + 16
    h = np.array(unpack('8H', d[header_offset:header_end]))
    # print(','.join('{:02x}'.format(x) for x in h))

    # extract information in global configurations
    frequency = np.int(h[1])
    current = np.int(h[3])
    gain = np.int(h[5])

    params = {'version': 0,
              'frequency': frequency,
              'current': current,
              'gain': gain}

    return params


def et3_header(d):
    """
    Parse the header of et files, bytes order:

    nVersion: int (4 Bytes)
    frame index: int (4 Bytes)
    time: double (8 bytes)
    tt: 3*double (reserved)
    -- offset 5 double (40 Bytes) --

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
    measure info: 5*16 int
    -- offset 90 int (360 Bytes)

    electrode SQ: 16 double
    reserved 14 double
    -- offset 30 double (240 Bytes)

    temperature 6 channel (double)
    reserved 16 Bytes
    -- offset 8 double (64 Bytes)

    total header = 40 + 320 + 360 + 240 + 64 = 1024
    """
    header_offset = 360
    header_end = header_offset + 40
    h = np.array(unpack('8I2f', d[header_offset:header_end]))

    # extract
    frequency = h[4]
    current = h[5]
    gain = h[6]

    # extract version info {et0: NA, et3: 1, shi: 3}
    version = int(unpack('I', d[:4])[0])
    # print('file version (header) = {}'.format(version))

    params = {'version': version,
              'frequency': frequency,
              'current': current,
              'gain': gain}

    return params


def gain_table(gain, current_in_ua):
    """
    Rescale data using if EIT is using programmable gain.

    pg table (in dll) used by Zhang Ge, 2017/12/06
    pgia_table = {0: 0.08,
                  1: 0.16,
                  2: 0.32,
                  3: 0.63,
                  4: 1.26,
                  5: 2.52,
                  6: 5.01,
                  7: 10.0}
    gain = 25.7 * keyvalue
    """
    # Programmable Gain table (scale mapping), see MainFrm.cpp
    pgia_table = {0: 4.112,
                  1: 8.224,
                  2: 16.448,
                  3: 32.382,
                  4: 64.764,
                  5: 129.528,
                  6: 257.514,
                  7: 514}

    # make sure gain is a valid key
    if gain not in pgia_table.keys():
        gain = 3

    # mapping ADC to resistor
    voltage_in_uv = 2.5 * 1000000.0 / 32768.0 / pgia_table[gain]
    scale = voltage_in_uv / current_in_ua

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
    file_name = '/data/dhca/dut/DATA.et3'
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
    df = et3.to_df()  # rel_date='2019/01/10'
    df['ati'] = np.abs(df).sum(axis=1) / 192.0

    # 3. plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(df.index.to_pydatetime(), df['ati'])
    ax.grid(True)

    # format time axis
    hfmt = dates.DateFormatter('%y/%m/%d %H:%M')
    ax.xaxis.set_major_formatter(hfmt)
    fig.autofmt_xdate()
    plt.show()

    return fig, ax


if __name__ == "__main__":
    demo()
