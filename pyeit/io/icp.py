# pylint: disable=no-member, invalid-name
""" Read ICP data (xlsx) """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
import re
import matplotlib.pyplot as plt
import pandas as pd


def load(fstr, resample=None):
    """
    load .xlsx file for ICP using pandas
    average all time stamp (resampling) in 'seconds' (default)
    return a new time series by pandas
    """
    # open .xlsx file
    xl_file = pd.ExcelFile(fstr)

    # parse only the first sheet (xlsx file has many sheets)
    df = xl_file.parse(xl_file.sheet_names[0])

    # using values to extract data from DataFrame
    icp_data = df["p"].values
    icp_timestmp = df["timestmp"].values
    # print('file %s size is %d' % (fstr, icp_data.size))

    # build a timeseries structure
    ts = pd.Series(icp_data, index=icp_timestmp)

    if resample is not None:
        # resample in 'seconds' as EIT system data rate
        ts = ts.resample(resample).mean()

    # return pandas series, you may use ts.index to extract time stamp
    # beware, there are np.NaN values in tss !
    return ts


# load in .csv format, much faster
def load_csv(fstr, resample=None):
    """
    load .xlsx file for ICP using pandas
    average all time stamp (resampling) in 'seconds' (default)

    Returns
    -------
    a new time series by pandas

    Notes
    -----
    to_datetime : ISO standard time format is XXXX-XX-XX DD:DD:DD.DDDD
    so as the format differs, loading performance drops.

    We may specify format,
    >>> t = pd.to_datetime(df['timestmp'], format="%Y/%m/%d %H:%M:%S.%f")
    however this is still slow before pandas-0.17.1

    instead, we apply a function row-wise on a series using map in DataFrame
    (However, you should really avoid apply with custom Python functions)
    it can work and works fine.
    """
    df = pd.read_csv(fstr)

    def f(x):
        """f = lambda x: re.sub('/', '-', x)"""
        return re.sub("/", "-", x)

    timestr = df["timestmp"].map(f)
    icp_timestmp = pd.to_datetime(timestr)
    icp_data = df["p"].values

    # build a timeseries structure
    ts = pd.Series(icp_data, index=icp_timestmp)

    if resample is not None:
        # averaged over 'seconds' as EIT systems
        ts = ts.resample(resample).mean()

    return ts


def convert(file_from, file_to, resample="s"):
    """
    load icp from xlsx and save it to a csv file,
    resample by default
    """
    if file_from.find("xlsx") > 0:
        ts = load(file_from, resample=resample)
    else:
        ts = load_csv(file_from, resample=resample)

    ts.to_csv(file_to)
    return ts


def demo_read_xlsx(filestr):
    """
    demo on how to load original (.xlsx) file
    """
    ts = load(filestr)
    return ts


def demo_read_csv(filestr):
    """
    export .xlsx in .csv and read using load_csv(), it is much faster
    """
    ts = load_csv(filestr, resample="600s")
    return ts


if __name__ == "__main__":
    ts_data = demo_read_csv("../../datasets/all.csv")

    # plot and see
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ts_data)
    ax.grid("on")
    plt.show()
