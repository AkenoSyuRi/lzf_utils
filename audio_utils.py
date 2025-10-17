import os
import wave
from pathlib import Path
from typing import Optional, Sequence, Union

import librosa
import numpy as np
import scipy
import soundfile


class AudioUtils:
    @staticmethod
    def ffmpeg_convert(in_audio_path, out_audio_path, sr=32000, nchannels=1, overwrite=True):
        extra_flags = "-y" if overwrite else "-n"
        extra_flags += " -v error"
        cmd = f"ffmpeg -i {in_audio_path} -ar {sr} -ac {nchannels} -acodec pcm_s16le {extra_flags} {out_audio_path}"
        ret = os.system(cmd)
        if ret == 0:
            print(out_audio_path)

    @staticmethod
    def apply_gain(sig, inc_db, rms_ref=1.0):
        """
        increase the volume of `sig` by `inc_db`
        :param sig: signal of floating point type in range [-1, 1]
        :param inc_db: the relative increment in db, can be positive or negative
        :param rms_ref: 1.0 for floating point
        :return: the new signal after volume up
        """
        assert np.issubdtype(sig.dtype, np.floating), "sig is not of floating-point type"

        cur_rms = np.sqrt(np.mean(sig**2))
        cur_db = 20 * np.log10(cur_rms / rms_ref)
        tar_db = cur_db + inc_db

        tar_rms = 10 ** (tar_db / 20)
        tar_sig = sig * tar_rms / cur_rms  # the signal is proportional to rms amplitude
        return tar_sig

    @staticmethod
    def save_to_mono(data, sr, base_dir, name):
        assert data.ndim == 1 or data.ndim == 2

        if data.ndim == 1:
            data = data.reshape(1, -1)
            channel_first = True
        else:
            channel_first = data.shape[0] < data.shape[1]

        if channel_first:
            n_channels = data.shape[0]
            if n_channels == 1:
                out_wav_path = os.path.join(base_dir, f"{name}.wav")
                soundfile.write(out_wav_path, data[0], sr)
            elif n_channels == 2:
                l_out_wav_path = os.path.join(base_dir, f"{name}_L.wav")
                r_out_wav_path = os.path.join(base_dir, f"{name}_R.wav")
                soundfile.write(l_out_wav_path, data[0], sr)
                soundfile.write(r_out_wav_path, data[1], sr)
            else:
                for i in range(n_channels):
                    out_wav_path = os.path.join(base_dir, f"{name}_chn{i}.wav")
                    soundfile.write(out_wav_path, data[i], sr)
        else:
            n_channels = data.shape[1]
            if n_channels == 1:
                out_wav_path = os.path.join(base_dir, f"{name}.wav")
                soundfile.write(out_wav_path, data[:, 0], sr)
            elif n_channels == 2:
                l_out_wav_path = os.path.join(base_dir, f"{name}_L.wav")
                r_out_wav_path = os.path.join(base_dir, f"{name}_R.wav")
                soundfile.write(l_out_wav_path, data[:, 0], sr)
                soundfile.write(r_out_wav_path, data[:, 1], sr)
            else:
                for i in range(n_channels):
                    out_wav_path = os.path.join(base_dir, f"{name}_chn{i}.wav")
                    soundfile.write(out_wav_path, data[:, i], sr)
        ...

    @staticmethod
    def save_to_segment(data, sr, win_len, win_sft, base_dir, name):
        data_len = len(data)
        if data_len < win_len:
            clip = np.pad(data, (0, win_len - data_len), mode="wrap")
            out_wav_path = os.path.join(base_dir, f"{name}_seg0.wav")
            soundfile.write(out_wav_path, clip, sr)
            print(out_wav_path)
            return

        for i, j in enumerate(range(0, data_len, win_sft)):
            out_wav_path = os.path.join(base_dir, f"{name}_seg{i}.wav")
            clip = data[j : j + win_len]

            nsamples = len(clip)
            if nsamples < win_len:
                if nsamples < (win_len // 4):
                    break
                clip = np.pad(clip, (0, win_len - nsamples), mode="wrap")
                # break

            soundfile.write(out_wav_path, clip, sr)
            print(out_wav_path)

    @staticmethod
    def data_generator(in_audio_path, frame_time, *, sr=None, ret_bytes=False):
        data, sr = librosa.load(in_audio_path, sr=sr)
        frame_len = int(sr * frame_time)

        for i in range(0, len(data), frame_len):
            clip = data[i : i + frame_len]
            if len(clip) == frame_len:
                if ret_bytes:
                    clip = (clip * 32768).astype(np.short)
                    yield clip.tobytes()
                else:
                    yield clip
        ...

    @staticmethod
    def wav_data_generator(in_audio_path, frame_time, *, sr=None, ret_bytes=False):
        assert in_audio_path.endswith(".wav"), "support wav format only"

        with wave.Wave_read(in_audio_path) as fp:
            if sr is None:
                sr = fp.getframerate()
            else:
                assert fp.getframerate() == sr
            assert fp.getnchannels() == 1
            assert fp.getsampwidth() == 2

            frame_len = int(sr * frame_time)

            desired_buff_len = frame_len * 2
            buff = fp.readframes(frame_len)
            while len(buff) == desired_buff_len:
                clip = np.frombuffer(buff, dtype=np.short)
                if ret_bytes:
                    yield clip.tobytes()
                else:
                    yield clip / 32768
                buff = fp.readframes(frame_len)

    @staticmethod
    def merge_channels(*data_list):
        n_channels = len(data_list)

        assert n_channels > 1

        max_len = 0
        for i in range(n_channels):
            assert data_list[i].ndim == 1
            max_len = max(data_list[i].shape[0], max_len)

        out_data = np.zeros((max_len, n_channels), dtype=data_list[0].dtype)
        for i in range(n_channels):
            data_len = data_list[i].shape[0]
            out_data[:data_len, i] = data_list[i]

        return out_data

    @staticmethod
    def pcm2wav(pcm_path, wav_path, sample_rate=32000, n_channels=1, sample_width=2):
        # assert not os.path.exists(wav_path)

        with open(pcm_path, "rb") as fp1, wave.Wave_write(wav_path) as fp2:
            raw_data = fp1.read()
            fp2.setsampwidth(sample_width)
            fp2.setnchannels(n_channels)
            fp2.setframerate(sample_rate)
            fp2.writeframes(raw_data)

    @staticmethod
    def wav2pcm(wav_path, pcm_path, overwrite=False):
        if not overwrite:
            assert not os.path.exists(pcm_path)

        with open(pcm_path, "wb") as fp1, wave.Wave_read(wav_path) as fp2:
            raw_data = fp2.readframes(fp2.getnframes())
            fp1.write(raw_data)

    @staticmethod
    def cal_rms_db(data, sr, frame_len=0.05, cal_total=False):
        def get_rms_db(data):
            rms = np.sqrt(np.mean(np.square(data)))
            db = 20 * np.log10(rms + 1e-7)
            return db

        if cal_total:
            db = get_rms_db(data)
            return db

        frame_len = int(sr * frame_len)
        remainder = len(data) % frame_len
        data = data[:-remainder].reshape(-1, frame_len) if remainder else data.reshape(-1, frame_len)
        db_list = []
        for i in range(data.shape[0]):
            frame = data[i]
            db = get_rms_db(frame)
            db_list.append(db)
        db = np.mean(db_list)
        return db


class BufferAdapter:
    def __init__(self, input_frame_len, output_frame_len):
        self.input_frame_len = input_frame_len
        self.output_frame_len = output_frame_len

        self.buf_len = np.lcm(input_frame_len, output_frame_len)
        self.buffer = np.zeros(self.buf_len)
        self.read_index = 0
        self.write_index = 0
        self.remain_size = 0

    def write(self, data_frame):
        assert len(data_frame) == self.input_frame_len
        self.buffer[self.write_index : self.write_index + self.input_frame_len] = data_frame

        self.remain_size += self.input_frame_len
        self.write_index += self.input_frame_len

        if self.write_index >= self.buf_len:
            self.write_index = 0

    def read(self):
        assert self.readable(), "you need to check readable before you read"
        data = self.buffer[self.read_index : self.read_index + self.output_frame_len]

        self.remain_size -= self.output_frame_len
        self.read_index += self.output_frame_len

        if self.read_index >= self.buf_len:
            self.read_index = 0

        return data

    def readable(self):
        return self.remain_size // self.output_frame_len > 0


class AudioReader:
    def __init__(self, in_audio_path_or_dir: Union[str, Path], sr=16000):
        self.in_audio_path_or_dir = Path(in_audio_path_or_dir)
        self.sr = sr

        if self.in_audio_path_or_dir.is_dir():
            self.in_audio_paths = sorted(self.in_audio_path_or_dir.glob("*.[wp][ac][vm]"))
            assert len(self.in_audio_paths) > 0, "no wav or pcm files found"
        else:
            self.in_audio_paths = [self.in_audio_path_or_dir]

        self.format = self.in_audio_paths[0].suffix.lower()
        assert self.format in [".wav", ".pcm"], "unsupported format: " + self.format

        self.in_fp_list = [self.open_audio_file(f) for f in self.in_audio_paths]
        ...

    def __del__(self):
        for fp in self.in_fp_list:
            fp.close()

    def open_audio_file(self, in_audio_path):
        if self.format == ".pcm":
            fp = open(in_audio_path, "rb")
        else:
            fp = wave.Wave_read(in_audio_path.as_posix())
        return fp

    def read_audio_data(self, frame_len):
        if self.format == ".pcm":
            read_len = frame_len * 2
            read_func_name = "read"
        else:
            read_len = frame_len
            read_func_name = "readframes"

        while True:
            is_complete = False
            mic_data_list = []
            for fp in self.in_fp_list:
                read_func = getattr(fp, read_func_name)
                raw_data = read_func(read_len)
                data = np.frombuffer(raw_data, dtype=np.short)
                is_complete = len(data) == frame_len
                data = data / 32768
                mic_data_list.append(data)

            if not is_complete:
                break

            yield np.stack(mic_data_list)


class AudioWriter:
    def __init__(self, out_wav_dir: Union[str, Path], sr, write_pcm=False):
        self.out_wav_dir = Path(out_wav_dir)
        self.sr = sr
        self.files_map = dict()
        self.closed = False
        self.write_pcm = write_pcm
        self.format = ".pcm" if write_pcm else ".wav"

        if not self.out_wav_dir.exists():
            self.out_wav_dir.mkdir()

    def _get_or_open(self, name_without_ext: str, data: np.ndarray):
        if name_without_ext in self.files_map:
            fp = self.files_map[name_without_ext]
        else:
            out_path = (self.out_wav_dir / name_without_ext).as_posix() + self.format
            if self.write_pcm:
                fp = open(out_path, "wb")
            else:
                channels = data.shape[1] if data.ndim == 2 else 1
                fp = wave.Wave_write(out_path)
                fp.setsampwidth(2)
                fp.setnchannels(channels)
                fp.setframerate(self.sr)
            self.files_map[name_without_ext] = fp
        return fp

    def _write_with_name(self, name, data, convert2short):
        write_func_name = "write" if self.write_pcm else "writeframes"
        fp = self._get_or_open(name, data)
        write_func = getattr(fp, write_func_name)
        if convert2short:
            write_func(self.to_short(data).tobytes())
        else:
            write_func(data.tobytes())

    def write_data_list(self, prefix, data_list, convert2short=True, onefile=False):
        if len(data_list) == 1 or onefile:
            data = np.array(data_list).squeeze().T
            self._write_with_name(prefix, data, convert2short)
        else:
            for i, data in enumerate(data_list):
                name = f"{prefix}_chn{i:02d}"
                self._write_with_name(name, data, convert2short)

    def _close(self):
        for fp in self.files_map.values():
            fp.close()
        self.closed = True

    def __del__(self):
        if not self.closed:
            self._close()
        ...

    @staticmethod
    def to_short(data):
        data = data * 32768
        np.clip(data, -32768, 32767, out=data)
        return data.astype(np.short)


class StreamingConvolution:

    def __init__(self, rir: np.ndarray, L=256):
        self.rir = rir
        self.L, self.M = L, len(rir)
        self.buffer = np.zeros(self.L + self.M - 1)

    def __call__(self, frame: np.ndarray):
        assert len(frame) == self.L, "frame length should be equal to L"
        conv_out = scipy.signal.fftconvolve(frame, self.rir)

        output = self.buffer[: self.L] + conv_out[: self.L]
        self.buffer[self.L :] += conv_out[self.L :]
        self.buffer[: self.M - 1] = self.buffer[self.L :]
        self.buffer[self.M - 1 :] = 0

        return output


class SignalGenerator:
    def __init__(
        self,
        sample_rate: int,
        amplitude: float = 0.8,
    ):
        self.sample_rate = sample_rate
        self.amplitude = amplitude
        self.signals = []
        ...

    def write_to(self, out_path: str, stereo=False):
        out_data = np.concatenate(self.signals)
        if stereo:
            out_data = np.column_stack([out_data, out_data])
        soundfile.write(out_path, out_data, self.sample_rate)
        self.signals.clear()
        print(f"Wrote {out_path}")
        ...

    def silence(self, duration: float):
        silence_signal = np.zeros(int(duration * self.sample_rate))
        self.signals.append(silence_signal)
        return self

    def pulse(self, duration: float, amplitude: Optional[float] = None):
        pulse_signal = np.zeros(int(duration * self.sample_rate))
        if amplitude is None:
            pulse_signal[0] = self.amplitude
        else:
            pulse_signal[0] = amplitude
        self.signals.append(pulse_signal)
        return self

    def chirp(self, duration: float, start_freq: float, end_freq: Optional[float] = None, linear=False):
        if end_freq is None:
            end_freq = self.sample_rate / 2

        sweep_signal = self.amplitude * librosa.chirp(
            fmin=start_freq, fmax=end_freq, sr=self.sample_rate, duration=duration, linear=linear
        )
        self.signals.append(sweep_signal)
        return self

    def sine_wave(self, duration: float, freqs: Union[float, Sequence[float]], interval=1.0):
        if isinstance(freqs, float):
            freqs = [freqs]
        t = np.arange(0, duration, 1 / self.sample_rate)

        sine_signal = self.amplitude * np.sin(2 * np.pi * freqs[0] * t)
        self.signals.append(sine_signal)

        for freq in freqs[1:]:
            self.silence(interval)
            sine_signal = self.amplitude * np.sin(2 * np.pi * freq * t)
            self.signals.append(sine_signal)
        return self
