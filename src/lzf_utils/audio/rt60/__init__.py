"""RT60 估计与诊断工具。

典型用法：

    from lzf_utils.audio import estimate_rt60, estimate_rt60_file

    result = estimate_rt60(rir, fs=16000, method="auto", bands="octave")
    result = estimate_rt60_file("room_rir.wav")

命令行：

    uv run rt60 room_rir.wav --bands octave --plot --output result.json
    uv run python -m lzf_utils.audio.rt60 room_rir.wav
"""

from __future__ import annotations

from typing import Any

from .estimate import (
    OCTAVE_BANDS_HZ,
    BandRT60Result,
    ChannelRT60Result,
    ChannelStats,
    MethodEstimate,
    NoiseInfo,
    RT60Estimate,
    RT60Result,
    estimate_rt60,
)
from .io import estimate_rt60_file

__all__ = [
    "OCTAVE_BANDS_HZ",
    "BandRT60Result",
    "ChannelRT60Result",
    "ChannelStats",
    "MethodEstimate",
    "NoiseInfo",
    "RT60Estimate",
    "RT60Result",
    "estimate_rt60",
    "estimate_rt60_file",
    "plot_rt60",
]


def plot_rt60(*args: Any, **kwargs: Any) -> Any:
    """惰性导入 matplotlib 的绘图入口。"""

    from .plot import plot_rt60 as _plot_rt60

    return _plot_rt60(*args, **kwargs)
