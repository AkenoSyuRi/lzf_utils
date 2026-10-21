"""RT60 的文件接口。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import soundfile

from .estimate import MethodOption, RT60Result, estimate_rt60


def estimate_rt60_file(
    path: str | Path,
    *,
    method: MethodOption = "auto",
    bands: Literal["octave", "none"] | Sequence[float] | None = None,
    detect_onset: bool = True,
    min_r2: float = 0.95,
    min_noise_margin_db: float = 10.0,
    max_method_disagreement: float = 0.25,
) -> RT60Result:
    """读取 WAV（或 soundfile 支持的其他格式）并估计 RT60。

    多通道文件的每一列按一个独立通道处理，不会先把各通道波形平均。
    """

    data, fs = soundfile.read(str(path), always_2d=True, dtype="float64")
    return estimate_rt60(
        data,
        fs=fs,
        method=method,
        bands=bands,
        time_axis=0,
        detect_onset=detect_onset,
        min_r2=min_r2,
        min_noise_margin_db=min_noise_margin_db,
        max_method_disagreement=max_method_disagreement,
    )
