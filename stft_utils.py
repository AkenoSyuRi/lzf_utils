import numpy as np


class Stft:
    def __init__(self, win_size, hop_size, in_channels, out_channels):
        self.win_size = win_size
        self.hop_size = hop_size
        self.overlap = win_size - hop_size
        self.fft_bins = win_size // 2 + 1

        self.window = np.hamming(win_size + 1)[1:]
        self.window /= self.window.sum()
        self.win_sum = self.get_win_sum_of_1frame(self.window, win_size, hop_size)

        self.in_win_data = np.zeros([in_channels, win_size])
        self.out_ola_data = np.zeros([out_channels, win_size])
        ...

    @staticmethod
    def get_win_sum_of_1frame(window, win_len, win_inc):
        assert win_len % win_inc == 0, "win_len must be equally divided by win_inc"
        win_square = window**2
        overlap = win_len - win_inc
        win_tmp = np.zeros(overlap + win_len)

        loop_cnt = win_len // win_inc
        for i in range(loop_cnt):
            win_tmp[i * win_inc : i * win_inc + win_len] += win_square
        win_sum = win_tmp[overlap : overlap + win_inc]
        assert (
            np.min(win_sum) > 0
        ), "the nonzero overlap-add constraint is not satisfied"
        return win_sum

    def transform(self, input_data):
        self.in_win_data[:, : self.overlap] = self.in_win_data[:, self.hop_size :]
        self.in_win_data[:, self.overlap :] = input_data

        spec_data = np.fft.rfft(self.in_win_data * self.window, axis=-1)
        return spec_data.squeeze()

    def inverse(self, input_spec):
        syn_data = np.fft.irfft(input_spec, axis=-1) * self.window

        self.out_ola_data += syn_data
        output_data = self.out_ola_data[:, : self.hop_size] / self.win_sum

        self.out_ola_data[:, : self.overlap] = self.out_ola_data[:, self.hop_size :]
        self.out_ola_data[:, self.overlap :] = 0
        return output_data.squeeze()
