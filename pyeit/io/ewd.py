# pylint: disable=no-member, invalid-name, too-many-instance-attributes
"""
load raw EWD into memory
"""
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
import os
from struct import unpack
import numpy as np


class EWD:
    """EIT raw/waveform data"""

    def __init__(self, file_name):
        """
        RAW data (.EWD) contains only data
        """
        self.file_name = file_name
        self.file_size = os.path.getsize(file_name)
        # 256 measurement waveforms, 128 points per wave, 2 bytes per point
        self.n_wave = 256
        self.n_point = 128
        self.n_data = self.n_wave * self.n_point
        self.frame_size = self.n_data * 2  # signed short
        self.n_frame = int(self.file_size / self.frame_size)
        self.tot_data = int(self.file_size / 2)

        raw = self.load_raw()
        self.wave = self.demodulate(raw)
        scale = 4096 / 65536 / 29.9 * 1000 / 1250
        self.data = (self.wave[:, :256] + 1j * self.wave[:, 256:]) * scale

    def load_raw(self):
        """load raw data"""
        raw = np.zeros((self.n_frame, self.n_data), dtype=np.int)
        with open(self.file_name, "rb") as fh:
            for i in range(self.n_frame):
                d = fh.read(self.frame_size)
                raw[i] = unpack("{}h".format(self.n_data), d)

        return raw

    def demodulate(self, raw):
        """demodulate raw data into [re, im]"""
        wave = np.zeros((self.n_frame, 2 * self.n_wave), dtype=np.double)
        sin_rom = np.sin(2.0 * np.pi * np.arange(self.n_point) / self.n_point)
        cos_rom = np.cos(2.0 * np.pi * np.arange(self.n_point) / self.n_point)
        for i in range(self.n_frame):
            dw = raw[i].reshape(self.n_wave, -1)
            wave_re = np.sum(dw * sin_rom, axis=1)
            wave_im = np.sum(dw * cos_rom, axis=1)
            wave[i] = np.concatenate([wave_re, wave_im]) / (self.n_point / 2.0)

        return wave

    def to_erd(self, src, dst):
        """combine ERD and demodulate EWD to a new file"""
        file_size = os.path.getsize(src)
        header_size = 1024
        data_size = 4096  # 2*256 doubles
        frame_size = header_size + data_size
        n_frame = int(file_size / frame_size)
        assert n_frame == self.n_frame
        with open(src, "rb") as fr, open(dst, "wb") as fw:
            for i in range(n_frame):
                h = fr.read(header_size)
                fr.read(data_size)  # skip data
                d = self.wave[i].tobytes()
                fw.write(h + d)
