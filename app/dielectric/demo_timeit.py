import re
import pandas as pd
import timeit

d_s = '2014-12-12 01:02:03.0030'
c_s = re.sub('-', '/', d_s)

d = pd.Series([d_s]*1000)
c = pd.Series([c_s]*1000)


def try1():
    pd.to_datetime(d)

t = timeit.timeit(try1, number=1000)
print(t*1000, 'us')


def try2():
    pd.to_datetime(c)

t = timeit.timeit(try2, number=1000)
print(t*1000, 'us')


def try3():
    pd.to_datetime(c, format='%Y/%m/%d %H:%M:%S.%f')

t = timeit.timeit(try3, number=1000)
print(t*1000, 'us')


def try4():
    pd.to_datetime(c, infer_datetime_format=True)

t = timeit.timeit(try4, number=1000)
print(t*1000, 'us')
