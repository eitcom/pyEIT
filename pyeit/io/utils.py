# pylint: disable=no-member, invalid-name
""" extract timestamp from file or path """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
import re
import pandas as pd


def string_to_time(fstr):
    """
    extract the start time from the path of .et3, .erd.
    """
    pattern = re.compile(r"(?P<date>\d+-\d+-\d+-\d+-\d+-\d+)")
    t = pattern.search(fstr)

    time_start = None
    if t is not None:
        time_start = t.group("date")
        # replace DATA2014-12-28-03-06-11 to DATA2014/12/28 03:06:11
        time_start = re.sub(r"-", r"/", time_start, count=2)
        time_start = re.sub(r"-", r" ", time_start, count=1)
        time_start = re.sub(r"-", r":", time_start)

    return time_start


def get_date_from_folder(file_str):
    """
    get datetime from file folder of .et3, .erd, i.e.,
    'DATA2015-01-29-16-57-30/', '2020-10-11-03-48-52/'
    """
    f = file_str.strip()
    if f[:-1] == "/":
        f = f[:-1]  # remove trailing '/'
    f = f.replace("DATA", "")
    # replace the 3rd occurrence of '-'
    w = [m.start() for m in re.finditer(r"-", f)][2]
    # before w do not change, after w, '-' -> ':'
    f = f[:w] + " " + f[w + 1 :].replace("-", ":")
    # now f becomes '2015-01-29 16:57:30'
    return pd.to_datetime(f)
