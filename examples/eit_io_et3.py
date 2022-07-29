# coding: utf-8
"""
demo for loading and play .et3 and .erd file

A sample EIT data will be uploaded.
"""
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from matplotlib import dates
import matplotlib.pyplot as plt
import numpy as np

from pyeit.io import ET3
import pyeit.eit.protocol as protocol

file_name = "/data/dhca/dut/DATA.et3"
# 1. using raw interface:
#    averaged transfer impedance from raw data
# et3_data = load(fstr, verbose=True)
# ts = load_time(fstr)
# ati = np.abs(et3_data).sum(axis=1)/192.0

# 2. using DataFrame interface:
#    's' is seconds (default)
#    'T' is minute
protocol_obj = protocol.create()
et3 = ET3(file_name, protocol_obj, reindex=True, verbose=False)
df = et3.to_df()  # rel_date='2019/01/10'
dp = et3.to_dp()
dp["ati"] = np.abs(df).sum(axis=1) / 192.0

# 3. plot
dp = dp[2000:]
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)
axt = ax.twinx()
ax.plot(dp.index, dp["t_left_ear"])
ax.plot(dp.index, dp["t_right_ear"])
ax.plot(dp.index, dp["t_naso"])
axt.plot(dp.index, dp["ati"])
ax.grid(True)
ax.legend(["Left ear", "Right ear", "Nasopharyngeal"])

hfmt = dates.DateFormatter("%y/%m/%d %H:%M")
ax.xaxis.set_major_formatter(hfmt)
fig.autofmt_xdate()
plt.show()
