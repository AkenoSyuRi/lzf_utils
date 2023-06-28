import numpy as np


class AudioUtils:
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

        cur_rms = np.sqrt(np.mean(sig ** 2))
        cur_db = 20 * np.log10(cur_rms / rms_ref)
        tar_db = cur_db + inc_db

        tar_rms = 10 ** (tar_db / 20)
        tar_sig = sig * tar_rms / cur_rms  # the signal is proportional to rms amplitude
        return tar_sig
