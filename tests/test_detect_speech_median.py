import importlib

import numpy as np
import pytest

module = importlib.import_module('lzf_utils.audio.detect_speech')


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('length', [1, 3, 5, 9])
def test_vectorized_median_preserves_shrink_edges(dtype, length):
    rng = np.random.default_rng(15)
    for size in [0, 1, 2, 4, 5, 6, 9, 133]:
        x = rng.normal(size=size).astype(dtype)
        for with_nan in [False, True]:
            if with_nan and size:
                x[size // 2] = np.nan
            radius = length // 2
            expected = np.array([
                np.median(x[max(0, i-radius):min(size, i+radius+1)])
                for i in range(size)
            ], dtype=dtype)
            actual = module._moving_median_shrink(x, length)
            np.testing.assert_array_equal(actual, expected)
            assert actual.dtype == expected.dtype
