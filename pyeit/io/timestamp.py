# pylint: disable=no-member, invalid-name
""" extract timestamp from file or path """

import re


def timestamp(fstr):
    """
    extract the start time from the path of .et3 file (strings)
    """
    pattern = re.compile(r"(?P<date>\d+-\d+-\d+-\d+-\d+-\d+)")
    t = pattern.search(fstr)

    time_start = None
    if t is not None:
        time_start = t.group('date')
        # replace DATA2014-12-28-03-06-11 to DATA2014/12/28 03:06:11
        time_start = re.sub(r"-", r"/", time_start, count=2)
        time_start = re.sub(r"-", r" ", time_start, count=1)
        time_start = re.sub(r"-", r":", time_start)

    return time_start
