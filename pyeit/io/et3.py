# pylint: disable=no-member, invalid-name, too-many-instance-attributes
"""
load .et0, .et3, .erd file into mem
using pack and unpack together with regular expression to filter out data.

These file types were developed by FMMU EIT group.
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


class ET3:
    """et3, erd file loader"""

    def __init__(
        self,
        file_name,
        data_type="auto",
        rel_date=None,
        fps=1,
        trim=True,
        verbose=False,
    ):
        """initialize file handler (supports .et0, .et3, .erd)"""
        self.file_name = file_name
        self.ext = splitext(file_name)[1][1:].lower()
        self.data_type = data_type
        self.rel_date = rel_date
        self.fps = fps
        self.trim = trim
        self.verbose = verbose

        # version info: et3 [<=3], erd [4]
        self.params = et_info(file_name, data_type)
        self.version = self.params["version"]
        self.rescale = self.params["rescale"]
        self.offset = self.params["offset"]
        self.nframe = self.params["nframe"]
        self.nadc = 8  # number of maximum ADC channels

        # frame = frame header + 2x256 (Re, Im) doubles frame data
        self.header_size = 1024  # Bytes
        self.data_num = 2 * 256  # doubles
        self.data_size = self.data_num * 8  # Bytes
        self.frame_size = self.header_size + self.data_size  # Bytes

        if verbose:
            print(self.ext, self.data_type, self.version)

        # load data (RAW data are complex-valued) and build datetime
        time_array, self.data, self.adc_array = self.load()
        self.ts = self.build_time(time_array)

    def load(self):
        """load a frame of data (header + data)"""
        time_array = np.zeros(self.nframe)
        x = np.zeros((self.nframe, self.data_num), dtype=np.double)
        adc_array = np.zeros((self.nframe, self.nadc), dtype=np.double)

        with open(self.file_name, "rb") as fh:
            fh.read(self.offset)

            for i in range(self.nframe):
                # get a frame
                d = fh.read(self.frame_size)
                # get time ticks
                time_array[i] = unpack("d", d[8:16])[0]
                # get ADC samples
                dp = d[960 : self.header_size]
                adc_array[i] = np.array(unpack("8d", dp))
                # get demodulated I,Q data
                x[i] = np.array(unpack("512d", d[self.header_size :]))

        # convert Re, Im to complex numbers
        raw_data = x[:, :256] + 1j * x[:, 256:]

        # remove all zeros columns in raw_data if trim is True
        if self.trim:
            idx = trim_pattern()
            data = raw_data[:, idx]
        else:
            data = raw_data

        # rescale if needed
        data = data * self.rescale

        return time_array, data, adc_array

    def build_time(self, time_array):
        """convert timestamp to datetime"""
        if self.version <= 3:  # .et0, .et3
            rel_date = "1899/12/30"  # excel format
            d_seconds = time_array * 86400  # convert days to seconds
        elif self.version == 4:  # .erd
            rel_date = "1970/01/01"  # posix format
            d_seconds = time_array / 1000 + 8 * 3600

        if self.rel_date is not None:  # mannual force datetime
            rel_date = self.rel_date
            d_seconds = np.arange(self.nframe) * 1.0 / self.fps

        d_seconds = np.round(d_seconds * 10.0) / 10.0
        ts = pd.to_datetime(rel_date) + pd.to_timedelta(d_seconds, "s")

        # check duplicated
        if any(ts.duplicated()):
            print("{}: duplicated index, dropped".format(self.file_name))
            print(ts[ts.duplicated()])

        return ts

    def to_df(self):
        """convert raw data to DataFrame"""
        df = pd.DataFrame(self.data, index=self.ts)
        df = df[~df.index.duplicated()]
        return df

    def to_dp(self, adc_filter=False):
        """convert raw 6 ADC channels data to DataFrame"""
        # left ear, right ear, Nasopharyngeal, rectal
        columns = ["tle", "tre", "tn", "tr", "c4", "c5", "c6", "c7"]
        dp = pd.DataFrame(self.adc_array, index=self.ts, columns=columns)

        if adc_filter:
            # filter auxillary sampled data
            # correct temperature (temperature can accidently be 0)
            dp.loc[dp["tle"] == 0, "tle"] = np.nan
            dp.loc[dp["tre"] == 0, "tre"] = np.nan
            dp.loc[dp["tn"] == 0, "tn"] = np.nan
            dp.loc[dp["tr"] == 0, "tr"] = np.nan

            dp.tle = med_outlier(dp.tle)
            dp.tre = med_outlier(dp.tre)
            dp.tn = med_outlier(dp.tn)
            dp.tr = med_outlier(dp.tr)

        dp = dp[~dp.index.duplicated()]
        return dp

    def to_csv(self):
        """
        save data in a .csv file

        this function is embedded in to pandas.DataFrame, simply call
        $ df.to_csv(file_to, columns=None, header=False, index=False)
        """
        raise NotImplementedError


def med_outlier(d, window=17):
    """ filter outliers using median filter """
    med = d.rolling(window, center=False).median()
    std = d.rolling(window, center=False).std()
    std[std == np.nan] = 0.0
    # replace med with d for outlier removal
    df = med[(d <= med + 3 * std) & (d >= med - 3 * std)]
    return df


def et_info(file_name, data_type):
    """
    Infer file-type and header information
    """
    with open(file_name, "rb") as fh:
        # move the cursor to the end (2) of the file
        fh.seek(0, 2)
        et3_len = fh.tell()

    # 'deprecated' version has extra file header (4096 Bytes)
    # 'new' : each frame is 5120 Bytes, no extra file header
    # 5120 = 1024 Bytes frame header + 256x2 Double frame data (4096 Bytes)
    is_header = (et3_len % 5120) != 0
    offset = 4096 if is_header else 0
    nframe = int((et3_len - offset) / 5120)

    parser = parse_header_et0 if data_type == "et0" else parse_header
    with open(file_name, "rb") as fh:
        fh.read(offset)  # skip offset
        d = fh.read(1024)  # get file info (header)
        params = parser(d)
    params["offset"] = offset
    params["nframe"] = nframe

    return params


def parse_header(d):
    """
    Parse the header of et files, bytes order:

    version: int (4 Bytes)
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
    # extract version info {et0: NA, et3: 1, shi: 3}
    version = int(unpack("I", d[:4])[0])
    h = np.array(unpack("8I2f", d[360:400]))
    frequency = h[4]
    current = h[5]
    gain = h[6]
    if version == 4:
        rescale = 4096 / 65536 / 29.9 * 1000 / 1250
    else:
        rescale = 1.0

    params = {
        "version": version,
        "frequency": frequency,
        "current": current,
        "gain": gain,
        "rescale": rescale,
    }

    return params


def parse_header_et0(d):
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

    hints on extract date from .et0 file.
    for .et0 file date is stored in string (bytes) format (utf-16, CJK)
    i.e., '2016Y01M01D 11H08M30S', offset=4, length=42 (21*2) Bytes
    """
    h = np.array(unpack("8H", d[48:64]))
    # print(','.join('{:02x}'.format(x) for x in h))

    # extract information in global configurations
    frequency = np.int(h[1])
    current = np.int(h[3])
    gain = np.int(h[5])
    rescale = gain_table(gain, current)  # convert voltage to Ohm

    params = {
        "version": 0,  # .et0
        "frequency": frequency,
        "current": current,
        "gain": gain,
        "rescale": rescale,
    }
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
    pgia_table = {
        0: 4.112,
        1: 8.224,
        2: 16.448,
        3: 32.382,
        4: 64.764,
        5: 129.528,
        6: 257.514,
        7: 514,
    }

    # make sure gain is a valid key
    if gain not in pgia_table.keys():
        gain = 3
    if current_in_ua <= 0 or current_in_ua > 1250:
        current_in_ua = 1000

    # mapping ADC to resistor
    voltage_in_uv = 2.5 * 1000000.0 / 32768.0 / pgia_table[gain]
    scale = voltage_in_uv / current_in_ua

    return scale


def trim_pattern():
    """
    Generate trim array (fixed)
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
    w = [m.start() for m in re.finditer(r"-", f)][2]
    # before w do not change, after w, '-' -> ':'
    f = f[:w] + " " + f[w + 1 :].replace("-", ":")
    # now f becomes '2015-01-29 16:57:30'
    return pd.to_datetime(f)


if __name__ == "__main__":
    # file_name = "/data/dhca/dut/DATA.et3"
    # file_name = "/data/dhca/subj086/data/DATA2017-06-14-18-33-57/RAWDATA.et0"
    # file_name = "/data/dhca/subj162/data/DATA2018-07-05-13-59-51/RAWDATA.et0"
    file_name = "/data/dhca/eh001case/data300/2020-10-29-15-36-05/EitRaw.ERD"

    # 1. using raw interface:
    #    averaged transfer impedance from raw data
    # et3_data = load(fstr, verbose=True)
    # ts = load_time(fstr)
    # ati = np.abs(et3_data).sum(axis=1)/192.0

    # 2. using DataFrame interface:
    #    's' is seconds (default)
    #    'T' is minute
    et3 = ET3(file_name, verbose=False)
    df = et3.to_df()  # rel_date='2019/01/10'
    dp = et3.to_dp()
    dp["ati"] = np.abs(df).sum(axis=1) / 192.0

    # 3. plot
    dp = dp[2000:]
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    axt = ax.twinx()
    ax.plot(dp.index, dp["tle"])
    ax.plot(dp.index, dp["tre"])
    ax.plot(dp.index, dp["tn"])
    axt.plot(dp.index, dp["ati"])
    ax.grid(True)
    ax.legend(["left", "right", "nt"])

    hfmt = dates.DateFormatter("%y/%m/%d %H:%M")
    ax.xaxis.set_major_formatter(hfmt)
    fig.autofmt_xdate()
