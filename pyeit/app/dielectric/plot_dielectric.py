# pylint: disable=no-member, invalid-name
"""load permitivity and conductivity from ITIS"""
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
import pandas as pd
import matplotlib.pyplot as plt

file_name = 'kidney'
file_str = './data/' + file_name + '.txt'

# print(open(file_str).read())
df = pd.read_csv(file_str, delim_whitespace=True, skiprows=2, header=None)
fat_freq = df[0] / 1000.0
fat_perm = df[1]
fat_cond = df[2]

# create twin plot
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# draw and label them
ax1.plot(fat_freq, fat_perm, 'b-', linewidth=2)
ax2.plot(fat_freq, fat_cond, 'r-', linewidth=2)

ax1.grid()
ax1.set_title(file_name)
ax1.set_xlabel('Frequency (KHz)')
ax1.set_ylabel('Permitivity', color='b')
ax2.set_ylabel('Electrical. Cond. (S/m)', color='r')

# export to pdf
fig.set_size_inches(6, 4)
fig.subplots_adjust(left=0.20, bottom=0.15, top=0.90, right=0.80)

plt.show()
# plt.savefig(file_save + '.pdf')
