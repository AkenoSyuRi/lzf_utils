import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import pyroomacoustics as pra
from numpy import ndarray

from audio_utils import AudioWriter


def convert_to_target_db(audio_data, target_db):
    if target_db > 0:
        target_db = -target_db

    # 将语音信号能量转化到TargetDb
    rms = np.sqrt(np.mean(audio_data**2, axis=-1))
    scalar = 10 ** (target_db / 20) / (rms + 1e-7)
    audio_data = audio_data * scalar.reshape(-1, 1)
    return audio_data


def get_audio_signal(audio_path, target_fs, target_db=None):
    data, fs = librosa.load(audio_path, sr=target_fs)
    if target_db:
        data = convert_to_target_db(data, target_db)
    return data


def get_azimuth_elevation(array_pos: ndarray, src_pos: ndarray, return_rad=False) -> Tuple[ndarray, ndarray]:
    """view array_pos as a reference point (origin of the coordinate),
    note: azi is same when abs(azi) == pi, remember to check this special case
    :param array_pos: (3,) the center coordinate of the mic array
    :param src_pos: (3) the coordinate of speakers
    :param return_rad: return the angle in radius or not(in degree)
    :param eps: a little no-zero number
    :return: azi shape (1,) range [-180,180], ele shape (1,) range [-90,90]
    """
    delta_d = src_pos - array_pos  # (3,)

    delta_x = delta_d[0]
    delta_y = delta_d[1]
    delta_z = delta_d[2]

    azi = np.arctan2(delta_y, delta_x)  # [-pi, pi]
    ele = np.arctan2(delta_z, np.sqrt(delta_x**2 + delta_y**2))  # [-pi/2, pi/2]

    if return_rad:
        return azi, ele

    azi = np.round(np.rad2deg(azi)).astype(int)
    ele = np.round(np.rad2deg(ele)).astype(int)

    print(f"src_pos: {src_pos}, az/el: {azi}/{ele}")
    return azi, ele


@dataclass
class SigInfo:
    sig: np.ndarray
    pos: np.ndarray
    delay: float = 0


class RoomDataSimulator:
    def __init__(self, room_size, mic_pos, fs, snr=None, rt60=None):
        """
        construct a RoomDataSimulator object
        :param room_size: (3,) in meters
        :param mic_pos: (3, num_mics) in meters
        :param fs: sampling rate in Hz
        :param snr: signal-to-noise ratio in dB
        :param rt60: reverberation time in seconds
        """
        self.room_size = room_size
        self.mic_pos = mic_pos
        self.fs = fs
        self.snr = snr
        self.rt60 = rt60

        self.sim_anechoic = rt60 is None  # simulate in anechoic room, no reverberation
        self.add_noise = snr is not None  # add gaussian noise
        self.acc_delay = None  # accumulated delay

        self.center_mic_coord = np.mean(mic_pos, -1)
        self.room = self._create_room()

        print(f"add_reverb={not self.sim_anechoic}, add_noise={self.add_noise}, {rt60=}, {snr=}")
        ...

    def _create_room(self):
        if self.sim_anechoic:
            e_absorption, max_order = 1.0, 0
        else:
            e_absorption, max_order = pra.inverse_sabine(self.rt60, self.room_size)
        room = pra.ShoeBox(
            self.room_size,
            fs=self.fs,
            materials=pra.Material(e_absorption),
            max_order=max_order,
        )
        room.add_microphone_array(pra.MicrophoneArray(self.mic_pos, room.fs))
        return room

    def map2sig_infos(self, in_wav_list, src_pos_list, delay: float = 0, add_delay=None):
        """
        construct the inputs for `simulate` function
        :param in_wav_list: source wav file list
        :param src_pos_list: source position list
        :param delay: delay in seconds, decide when to begin the simulation
        :param add_delay: accumulate the delay prepare for the next invoking
        :return: sig_info list
        """
        assert len(in_wav_list) > 0 and len(in_wav_list) == len(src_pos_list), "invalid inputs"

        if self.acc_delay is None:
            self.acc_delay = delay

        sig_infos = []
        for in_wav, src_pos in zip(in_wav_list, src_pos_list):
            if isinstance(in_wav, np.ndarray):
                data = in_wav
            else:
                data, _ = librosa.load(in_wav, sr=self.fs)
            get_azimuth_elevation(self.center_mic_coord, src_pos)
            sig_infos.append(SigInfo(data, src_pos, delay=self.acc_delay))

            if add_delay:
                self.acc_delay += add_delay

        return sig_infos

    def simulate(self, *sig_infos: SigInfo, random_seed=0):
        assert len(sig_infos) > 0, "no input signals"

        for sig_info in sig_infos:
            self.room.add_source(sig_info.pos, sig_info.sig, sig_info.delay)

        if self.add_noise:
            np.random.seed(random_seed)
            self.room.simulate(snr=self.snr)
        else:
            self.room.simulate()
        ...

    def save(
        self,
        out_dir: str | Path,
        out_name: str,
        out_db: float = None,
        mono=True,
        save_pcm=False,
        audio_writer: AudioWriter = None,
    ):
        if self.sim_anechoic:
            out_name += "_anechoic"
        else:
            out_name += f"_rt60_{self.rt60:.1f}s"
        if self.add_noise:
            out_name += f"_snr{self.snr}"

        if out_db is not None:
            out_sig = convert_to_target_db(self.room.mic_array.signals, out_db)
        else:
            out_sig = self.room.mic_array.signals

        if audio_writer is not None:
            aw = audio_writer
        else:
            out_dir = Path(out_dir) / out_name
            shutil.rmtree(out_dir, ignore_errors=True)
            aw = AudioWriter(out_dir, self.fs, save_pcm)

        if mono:
            aw.write_data_list(out_name, out_sig)
        else:
            aw.write_data_list(out_name, out_sig, onefile=True)

        return aw
