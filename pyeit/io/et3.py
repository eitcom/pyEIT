# pylint: disable=no-member, invalid-name, too-many-instance-attributes
"""
load .et0, .et3, .erd file into mem
using pack and unpack together with regular expression to filter out data.

These file types were developed by FMMU EIT group.
Please cite the following paper if you are using et3 in your research:
    Fu, Feng, et al. "Use of electrical impedance tomography to monitor
    regional cerebral edema during clinical dehydration treatment."
    PloS one 9.12 (2014): e113202.

2022-05-16: FMMU ET3 using [m, n](i.e., V[m] - V[n]) not [n, m],
            which adds a negative sign to make it compatible.
"""
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
import warnings
import os
from struct import unpack
import numpy as np
import pandas as pd


class ET3:
    """et3, erd file loader"""

    def __init__(
        self,
        file_name,
        protocol,
        data_type="auto",
        rel_date=None,
        fps=1,
        reindex=False,
        meas_current=False,
        verbose=False,
    ):
        """
        initialize file handler (supports .et0, .et3, .erd)

        Parameters
        ----------
        file_name : String
            file path for EIT data.
        protocol : PyEITProtocol
            a protocol dataclass specify the interpretation of this file format
        data_type : String, optional
            manually set data type i.e., "et0". The default is "auto".
        rel_date : Datetime, optional
            Datetime, i.e., "2014/01/25". If specified, the time information
            in header is ignored. The default is None.
        fps : Int, optional
            Frame per second, valid when rel_date specified. The default is 1.
        reindex : Bool, optional
            reindex ERD data order to ET3 format (RALP -> LARP) before trim.
        meas_current : Bool, optional
            Keep Measurements on current carry electrodes.
        verbose : Bool, optional
            Print debuging information. The default is False.
        """
        self.file_name = file_name
        self.ext = os.path.splitext(file_name)[1][1:].lower()
        self.rel_date = rel_date
        self.fps = fps
        self.reindex = reindex
        self.meas_current = meas_current
        self.verbose = verbose

        # frame = frame header + 2x256 (Re, Im) doubles frame data
        self.header_size = 1024  # Bytes
        self.n_data = 2 * 256  # doubles
        self.data_size = self.n_data * 8  # Bytes
        self.frame_size = self.header_size + self.data_size  # Bytes
        self.nadc = 8  # number of maximum ADC channels

        file_size = os.path.getsize(file_name)
        offset = file_size % self.frame_size
        # 'deprecated' file type has an extra file header (4096 Bytes)
        # 'et' : each frame is 5120 Bytes, no extra file header
        # 5120 = 1024 Bytes frame header + 256x2 Double frame data (4096 Bytes)
        if not ((offset == 0) or (offset == 4096)):
            raise ValueError("Wrong File Offset = {}".format(offset))
        n_frame = int((file_size - offset) / self.frame_size)
        self.offset = offset
        self.n_frame = n_frame

        # extract system configuration
        self.p = self.setup(data_type)

        # load data (RAW data are complex-valued)
        time_array, data, self.adc_array = self.load()

        # build timeseries index: datetime
        self.ts = self.build_ts(time_array)

        # reindex ERD format to FMMU ET format
        if self.p["data_format"] == "erd":
            if self.reindex is True:
                ind = self.erd2et()
                data = data[:, ind]
            else:
                warnings.warn("File format is ERD but reindex to ET3 is set to False")
        # remove all measurements on current carring electrodes
        if not self.meas_current:
            data = data[:, protocol.keep_ba]
        self.data = data

    def setup(self, data_type):
        """
        Infer file-type and header information
        """
        # parse header
        parser = parse_header_et0 if data_type == "et0" else parse_header
        with open(self.file_name, "rb") as fh:
            fh.read(self.offset)  # skip offset
            d = fh.read(self.header_size)
            p = parser(d)  # extract information from frame header

        # default et3: version <= 3 and ext in ["et0", "et3"]
        data_format = "et"
        date0 = "1899/12/30"  # excel format
        ts_scale = 86400  # convert days to seconds
        ts_offset = 0
        scale = 1.0
        # other formats
        if data_type == "et0":  # et0: convert voltage to Ohm
            scale = gain_table(p["gain"], p["current"])
        if p["version"] == 4 or self.ext == "erd":
            data_format = "erd"
            date0 = "1970/01/01"  # erd: posix format
            ts_scale = 0.001
            ts_offset = 8 * 3600
            scale = 4096 / 65536 / 29.9 * 1000 / 1250  # fixed gain

        p_new = {
            "data_format": data_format,
            "date0": date0,
            "ts_scale": ts_scale,
            "ts_offset": ts_offset,
            "scale": scale,
        }
        p.update(p_new)

        return p

    @staticmethod
    def erd2et():
        """
        CT scans RALP, Starting at the 9 o’clock position and moving clockwise
        in 90 degree intervals, we are looking at the
        Right, Anterior, Left and Posterior aspects of the patient.

        ERD electrodes: 0 (R) 4 (A) 8 (L) 12 (P)
        ET electrodes : 0 (L) 4 (A) 8 (R) 12 (P)

        These systems are using opposition stimulation (a, b) = (0-8), (1-9), ..
        and rotate measurements a + (1-0, 2-1, .., 15-14).
        """
        erd_electrodes = np.arange(16)  # 0 .. 15
        et3_electrodes = np.hstack([np.arange(9, 0, -1), np.arange(16, 9, -1)]) - 1
        row_swap = [np.where(i == et3_electrodes)[0][0] for i in erd_electrodes]
        ind = np.arange(256).reshape(16, -1)
        reind = np.fliplr(ind[row_swap, :]).reshape(-1)  # CCW to CW
        return reind

    def load(self):
        """load frames (header + data)"""
        time_array = np.zeros(self.n_frame, dtype=np.double)
        x = np.zeros((self.n_frame, self.n_data), dtype=np.double)
        adc_array = np.zeros((self.n_frame, self.nadc), dtype=np.double)
        with open(self.file_name, "rb") as fh:
            fh.read(self.offset)
            for i in range(self.n_frame):
                # get a whole frame
                d = fh.read(self.frame_size)
                # extract time ticks
                time_array[i] = unpack(self.p["ts_format"], d[8:16])[0]
                # extract ADC samples (double precision)
                dp = d[960 : self.header_size]
                adc_array[i] = np.array(unpack("8d", dp))
                # extract demodulated I,Q data
                x[i] = np.array(unpack("512d", d[self.header_size :]))

        # build complex-valued data from Re, Im measurements
        data = x[:, :256] + 1j * x[:, 256:]

        # rescale data to Ohms
        data = -data * self.p["scale"]

        return time_array, data, adc_array

    def build_ts(self, time_array):
        """convert delta time (seconds) to datetime series"""
        if self.rel_date is not None:  # user sets datetime
            rel_date = self.rel_date
            d_seconds = np.arange(self.n_frame) * 1.0 / self.fps
        else:
            rel_date = self.p["date0"]
            d_seconds = time_array * self.p["ts_scale"] + self.p["ts_offset"]

        d_seconds = np.round(d_seconds * 1000.0) / 1000.0  # 1 ms resolution
        ts = pd.to_datetime(rel_date) + pd.to_timedelta(d_seconds, "s")

        # check duplicated
        if any(ts.duplicated()):
            print("{}: duplicated datetime:".format(self.file_name))
            print(ts[ts.duplicated()])

        return ts

    def to_df(self):
        """convert raw EIT data to DataFrame"""
        df = pd.DataFrame(self.data, index=self.ts)
        df = df[~df.index.duplicated()]
        return df

    def to_dp(self, adc_filter=False, window=17):
        """
        in new ET3 data, the left ear, right ear, Nasopharyngeal, rectal
        temperature are recorded in the headers of .et3 file.
        This script convert raw ADC data to DataFrame.
        """
        #
        columns = [
            "t_left_ear",
            "t_right_ear",
            "t_naso",
            "t_renal",
            "aux_c4",
            "aux_c5",
            "aux_c6",
            "aux_c7",
        ]
        dp = pd.DataFrame(self.adc_array, index=self.ts, columns=columns)
        dp = dp[~dp.index.duplicated()]

        if adc_filter:
            # correct temperature (temperature can accidently be 0)
            # filter auxillary sampled data
            for c in columns:
                dp.loc[dp[c] == 0, c] = np.NAN
                dp[c] = med_outlier(dp[c], window=window)

        return dp

    def to_csv(self):
        """
        save data in a .csv file

        this function is embedded in to pandas.DataFrame, simply call
        $ df.to_csv(file_to, columns=None, header=False, index=False)
        """
        raise NotImplementedError


def med_outlier(d, window=17, delta=3):
    """filter outliers using median filter [abs difference of median]"""
    med = d.fillna(method="ffill").rolling(window, center=True).median()
    diff = np.abs(d - med)
    diff_med = np.nanmedian(diff) + 1e-8
    mask = (diff / diff_med) > delta
    # replace med with d for outlier removal
    d[mask] = med[mask]
    return d


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
    # extract version info {et0: NA, et3: 1, shi: 3, erd: 4}
    version = int(unpack("I", d[:4])[0])
    # extract timestamp
    timestamp = unpack("H", d[14:16])[0]
    if timestamp == 0:  # higher 8 bits are 0s
        ts_format = "Q"  # d[8:16] is inferred as Uint64
    else:
        ts_format = "d"  # d[8:16] is double format
    # extract configurations
    h = np.array(unpack("8I2f", d[360:400]))
    frequency = h[4]
    current = h[5]
    gain = h[6]

    p = {
        "ts_format": ts_format,
        "version": version,
        "frequency": frequency,
        "current": current,
        "gain": gain,
    }
    return p


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

    p = {
        "ts_format": "c",
        "version": 0,  # .et0
        "frequency": frequency,
        "current": current,
        "gain": gain,
    }
    return p


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
    Deprecated, use protocol.keep_ba to trim the data.
    Generate trim array (masked value, 0) on rotating measurements,
    0......00......0 0......00......0 .. 0......00......0
    """
    idx = np.ones(256, dtype=np.bool)

    for i in range(32):
        j = i * 8
        # exclude the stimulation indices (i.e., 0-8 stimulus)
        idx[j] = False
        idx[j + 7] = False

    return idx
