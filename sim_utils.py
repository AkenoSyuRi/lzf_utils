import shutil
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pyroomacoustics as pra
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


def cal_source_direction(center_coord, src_pos):
    x = src_pos[0] - center_coord[0]
    y = src_pos[1] - center_coord[1]
    if x == 0 and y == 0:
        cos_az = 1
    else:
        cos_az = x / np.sqrt(x**2 + y**2)
    azimuth = round(np.rad2deg(np.arccos(cos_az)), 3)
    if y < 0:
        azimuth = 360 - azimuth

    z_diff = abs(src_pos[2] - center_coord[2])
    dis = np.sqrt(np.sum(np.square(center_coord - src_pos)))
    cos_el = z_diff / dis
    elevation = round(np.rad2deg(np.arccos(cos_el)), 1)

    print(f"src_pos: {src_pos}, az/el: {azimuth:.1f}/{elevation:.1f}")
    return azimuth, elevation


@dataclass
class SigInfo:
    sig: np.ndarray
    pos: np.ndarray | list
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

        print(
            f"add_reverb={not self.sim_anechoic}, add_noise={self.add_noise}, {rt60=}, {snr=}"
        )
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

    def map2sig_infos(
        self, in_wav_list, src_pos_list, delay: float = 0, add_delay=None
    ):
        """
        construct the inputs for `simulate` function
        :param in_wav_list: source wav file list
        :param src_pos_list: source position list
        :param delay: delay in seconds, decide when to begin the simulation
        :param add_delay: accumulate the delay prepare for the next invoking
        :return: sig_info list
        """
        assert len(in_wav_list) > 0 and len(in_wav_list) == len(
            src_pos_list
        ), "invalid inputs"

        if self.acc_delay is None:
            self.acc_delay = delay

        sig_infos = []
        for in_wav, src_pos in zip(in_wav_list, src_pos_list):
            if isinstance(in_wav, np.ndarray):
                data = in_wav
            else:
                data, _ = librosa.load(in_wav, sr=self.fs)
            cal_source_direction(self.center_mic_coord, src_pos)
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
