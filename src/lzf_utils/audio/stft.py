from typing import Union, Optional

import numpy as np
from scipy import signal


class Stft:
    def __init__(
        self,
        fft_size: int,
        win_size: Optional[int] = None,
        hop_size: Optional[int] = None,
        in_channels: int = 1,
        out_channels: int = 1,
        window: Union[str, np.ndarray] = "hann",
    ):
        assert fft_size >= win_size, "fft_size must be greater than or equal to win_size"

        self.fft_size = fft_size

        if win_size is None:
            win_size = fft_size
        self.win_size = win_size

        if hop_size is None:
            hop_size = win_size // 2
        self.hop_size = hop_size

        self.overlap = win_size - hop_size
        self.fft_bins = win_size // 2 + 1

        if isinstance(window, str):
            self.window = signal.get_window(window, win_size, fftbins=True)
        else:
            assert len(window) == win_size, "window size must be equal to win_size"
            self.window = window
        self.win_sum = self.get_win_sum_of_1frame(self.window, win_size, hop_size)

        self.in_win_data = np.zeros([in_channels, win_size])
        self.out_ola_data = np.zeros([out_channels, win_size])
        ...

    @staticmethod
    def get_win_sum_of_1frame(window, win_len, win_inc):
        assert win_len % win_inc == 0, "win_len must be equally divided by win_inc"
        win_square = window**2
        win_sum = np.zeros(win_inc)

        loop_cnt = win_len // win_inc
        for i in range(loop_cnt):
            win_sum += win_square[i * win_inc : (i + 1) * win_inc]
        assert np.min(win_sum) > 0, "the nonzero overlap-add constraint is not satisfied"
        return win_sum

    def transform(self, input_data):
        self.in_win_data[:, : self.overlap] = self.in_win_data[:, self.hop_size :]
        self.in_win_data[:, self.overlap :] = input_data

        spec_data = np.fft.rfft(self.in_win_data * self.window, n=self.fft_size, axis=-1)
        return spec_data.squeeze()

    def inverse(self, input_spec):
        syn_data = np.fft.irfft(input_spec, n=self.fft_size, axis=-1)[..., : self.win_size]

        self.out_ola_data += syn_data * self.window
        output_data = self.out_ola_data[:, : self.hop_size] / self.win_sum

        self.out_ola_data[:, : self.overlap] = self.out_ola_data[:, self.hop_size :]
        self.out_ola_data[:, self.overlap :] = 0
        return output_data.squeeze()
