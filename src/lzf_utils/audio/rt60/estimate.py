"""基于 RIR 的 RT60 估计。

核心流程：

1. 可选地检测直达声起点，去掉前面的无效静音；
2. 用 Lundeby 迭代法估计背景噪声功率，并确定 Schroeder 积分的有效终点；
3. 对截断后的 RIR 做反向能量积分，得到 EDC；
4. 在 EDT / T20 / T30 各自区间内做最小二乘直线拟合，外推 RT60；
5. 用衰减范围、拟合优度、斜率、噪声余量和 T20/T30 一致性共同判断结果是否可信。

本模块只依赖 NumPy / SciPy，不导入 matplotlib。文件读写与绘图分别在
``lzf_utils.audio.rt60.io`` 和 ``lzf_utils.audio.rt60.plot`` 中。
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal

FloatArray = NDArray[np.float64]
MethodName = Literal["EDT", "T20", "T30"]
MethodOption = Literal["auto", "EDT", "T20", "T30"]

OCTAVE_BANDS_HZ: tuple[float, ...] = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0)

_METHOD_RANGES: dict[MethodName, tuple[float, float]] = {
    "EDT": (0.0, -10.0),
    "T20": (-5.0, -25.0),
    "T30": (-5.0, -35.0),
}
_METHODS: tuple[MethodName, ...] = ("EDT", "T20", "T30")
_TINY = np.finfo(np.float64).tiny
_MIN_DURATION_S = 0.02
_FIT_RANGE_TOLERANCE_DB = 1.0
_BAND_ENERGY_FLOOR_DB = -50.0
_MAX_METHOD_DISAGREEMENT = 0.25


@dataclass(frozen=True)
class MethodEstimate:
    """一种估计方法（EDT / T20 / T30）的拟合结果。"""

    method: MethodName
    rt60: float
    valid: bool
    reason: str | None
    r2: float
    slope_db_per_s: float
    fit_start_s: float
    fit_end_s: float
    fit_start_db: float
    fit_end_db: float
    fit_range_db: float
    n_samples: int
    noise_contamination_db: float
    noise_margin_db: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "rt60": _finite_or_none(self.rt60),
            "valid": self.valid,
            "reason": self.reason,
            "r2": _finite_or_none(self.r2),
            "slope_db_per_s": _finite_or_none(self.slope_db_per_s),
            "fit_start_s": _finite_or_none(self.fit_start_s),
            "fit_end_s": _finite_or_none(self.fit_end_s),
            "fit_start_db": _finite_or_none(self.fit_start_db),
            "fit_end_db": _finite_or_none(self.fit_end_db),
            "fit_range_db": _finite_or_none(self.fit_range_db),
            "n_samples": self.n_samples,
            "noise_contamination_db": _finite_or_none(self.noise_contamination_db),
            "noise_margin_db": _finite_or_none(self.noise_margin_db),
        }


@dataclass(frozen=True)
class NoiseInfo:
    """背景噪声估计与积分截断诊断。

    ``noise_power`` 是 RIR 先按峰值样本归一化到 1 后的均方噪声功率；
    ``noise_floor_db`` 是把该噪声功率在有效积分区间内累积成 EDC 后的
    保守噪声底，单位为 dB（相对 EDC[0]）。
    """

    algorithm: str
    noise_power: float
    peak_snr_db: float
    noise_floor_db: float
    truncation_time_s: float
    truncated_samples: int
    truncated: bool
    smoothing_time_s: float
    iterations: int
    converged: bool
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "noise_power": _finite_or_none(self.noise_power),
            "peak_snr_db": _finite_or_none(self.peak_snr_db),
            "noise_floor_db": _finite_or_none(self.noise_floor_db),
            "truncation_time_s": _finite_or_none(self.truncation_time_s),
            "truncated_samples": self.truncated_samples,
            "truncated": self.truncated,
            "smoothing_time_s": _finite_or_none(self.smoothing_time_s),
            "iterations": self.iterations,
            "converged": self.converged,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RT60Estimate:
    """单个信号（一个通道或一个频带）的 RT60 估计。"""

    edt: float
    t20: float
    t30: float
    rt60: float
    method_used: MethodName | None
    r2: float
    valid: bool
    reason: str | None
    decay_range_db: float
    onset_sample: int
    onset_time_s: float
    truncated_samples: int
    band_energy_db: float | None
    methods: dict[MethodName, MethodEstimate]
    noise: NoiseInfo
    edc_time_s: FloatArray = field(repr=False)
    edc_db: FloatArray = field(repr=False)

    def to_dict(self, include_edc: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "edt": _finite_or_none(self.edt),
            "t20": _finite_or_none(self.t20),
            "t30": _finite_or_none(self.t30),
            "rt60": _finite_or_none(self.rt60),
            "method_used": self.method_used,
            "r2": _finite_or_none(self.r2),
            "valid": self.valid,
            "reason": self.reason,
            "decay_range_db": _finite_or_none(self.decay_range_db),
            "onset_sample": self.onset_sample,
            "onset_time_s": _finite_or_none(self.onset_time_s),
            "truncated_samples": self.truncated_samples,
            "band_energy_db": _finite_or_none(self.band_energy_db),
            "methods": {name: item.to_dict() for name, item in self.methods.items()},
            "noise": self.noise.to_dict(),
        }
        if include_edc:
            data["edc_time_s"] = [_finite_or_none(v) for v in self.edc_time_s]
            data["edc_db"] = [_finite_or_none(v) for v in self.edc_db]
        return data


@dataclass(frozen=True)
class ChannelStats:
    """跨通道统计；无效通道不参与 median / mean / std。"""

    n_valid: int
    n_total: int
    median: float
    mean: float
    std: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_valid": self.n_valid,
            "n_total": self.n_total,
            "median": _finite_or_none(self.median),
            "mean": _finite_or_none(self.mean),
            "std": _finite_or_none(self.std),
        }


@dataclass(frozen=True)
class BandRT60Result:
    """一个倍频程频带在所有通道上的结果。"""

    center_frequency_hz: float
    channels: tuple[RT60Estimate, ...]
    stats: ChannelStats

    def to_dict(self, include_edc: bool = False) -> dict[str, Any]:
        return {
            "center_frequency_hz": float(self.center_frequency_hz),
            "stats": self.stats.to_dict(),
            "channels": [item.to_dict(include_edc=include_edc) for item in self.channels],
        }


@dataclass(frozen=True)
class ChannelRT60Result:
    """一个通道的全频带与分频带结果。"""

    channel: int
    broadband: RT60Estimate
    bands: dict[float, RT60Estimate]

    @property
    def rt60(self) -> float:
        return self.broadband.rt60

    @property
    def edt(self) -> float:
        return self.broadband.edt

    @property
    def t20(self) -> float:
        return self.broadband.t20

    @property
    def t30(self) -> float:
        return self.broadband.t30

    @property
    def method_used(self) -> MethodName | None:
        return self.broadband.method_used

    @property
    def valid(self) -> bool:
        return self.broadband.valid

    @property
    def reason(self) -> str | None:
        return self.broadband.reason

    def to_dict(self, include_edc: bool = False) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "broadband": self.broadband.to_dict(include_edc=include_edc),
            "bands": {
                _freq_label(freq): item.to_dict(include_edc=include_edc)
                for freq, item in self.bands.items()
            },
        }


@dataclass(frozen=True)
class RT60Result:
    """一次 RT60 分析的完整结果。"""

    fs: float
    method: MethodOption
    channels: tuple[ChannelRT60Result, ...]
    bands_hz: tuple[float, ...]
    bands: dict[float, BandRT60Result]
    stats: ChannelStats
    method_stats: dict[MethodName, ChannelStats]
    rir: FloatArray = field(repr=False)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def band_stats(self) -> dict[float, ChannelStats]:
        return {freq: band.stats for freq, band in self.bands.items()}

    def channel(self, index: int) -> ChannelRT60Result:
        """按通道号取结果。"""

        return self.channels[index]

    def to_dict(self, include_edc: bool = False) -> dict[str, Any]:
        return {
            "fs": self.fs,
            "method": self.method,
            "n_channels": self.n_channels,
            "stats": self.stats.to_dict(),
            "method_stats": {name: stats.to_dict() for name, stats in self.method_stats.items()},
            "bands_hz": [float(freq) for freq in self.bands_hz],
            "band_stats": {_freq_label(freq): stats.to_dict() for freq, stats in self.band_stats.items()},
            "bands": {
                _freq_label(freq): band.to_dict(include_edc=include_edc)
                for freq, band in self.bands.items()
            },
            "channels": [channel.to_dict(include_edc=include_edc) for channel in self.channels],
        }

    def to_json(
        self,
        path: str | Path | None = None,
        *,
        include_edc: bool = False,
        indent: int = 2,
    ) -> str:
        """序列化为严格 JSON（非有限浮点数写为 ``null``）。"""

        text = json.dumps(
            self.to_dict(include_edc=include_edc),
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
        )
        if path is not None:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text


@dataclass
class _NoiseResult:
    ok: bool
    noise_power: float
    peak_snr_db: float
    crossing_s: float
    block_s: float
    iterations: int
    converged: bool
    reason: str | None


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _freq_label(freq: float) -> str:
    number = float(freq)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def _compute_stats(values: Sequence[float], n_total: int) -> ChannelStats:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return ChannelStats(
            n_valid=0,
            n_total=n_total,
            median=math.nan,
            mean=math.nan,
            std=math.nan,
        )
    std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    return ChannelStats(
        n_valid=int(finite.size),
        n_total=n_total,
        median=float(np.median(finite)),
        mean=float(np.mean(finite)),
        std=std,
    )


def _detect_onset(x: FloatArray, fs: float) -> int:
    """用双阈值在采样级找到直达声起点。

    先找第一段连续超过 ``peak - 25 dB`` 的样本，再向回扩展到
    ``peak - 50 dB``（最多 50 ms），避免把较弱的直达声或早期反射切掉。
    不使用毫秒级前置补偿，避免在 EDT 拟合区间里引入平坦段。
    """

    magnitude = np.abs(x)
    peak = float(np.max(magnitude))
    if peak <= 0.0:
        return 0

    high = peak * 10.0 ** (-25.0 / 20.0)
    low = peak * 10.0 ** (-50.0 / 20.0)

    onset: int | None = None
    run_start = 0
    run_length = 0
    for index, value in enumerate(magnitude):
        if value >= high:
            if run_length == 0:
                run_start = index
            run_length += 1
            if run_length >= 2:
                onset = run_start
                break
        else:
            run_length = 0

    if onset is None:
        onset = int(np.argmax(magnitude))

    lookback = max(1, round(0.05 * fs))
    lower_bound = max(0, onset - lookback)
    while onset > lower_bound and magnitude[onset - 1] >= low:
        onset -= 1

    return int(onset)


def _smooth_energy(energy: FloatArray, block: int, fs: float) -> tuple[FloatArray, FloatArray]:
    block = max(1, int(block))
    n_blocks = energy.size // block
    if n_blocks < 2:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    data = energy[: n_blocks * block].reshape(n_blocks, block).mean(axis=1)
    time = (np.arange(n_blocks, dtype=np.float64) + 0.5) * block / fs
    return data, time


def _fit_line(time: FloatArray, values: FloatArray) -> tuple[float, float, float]:
    if time.size < 2:
        return math.nan, math.nan, math.nan
    matrix = np.vstack((time, np.ones_like(time))).T
    (slope, intercept), *_ = np.linalg.lstsq(matrix, values, rcond=None)
    residual = values - (slope * time + intercept)
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum((values - values.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return float(slope), float(intercept), float(r2)


def _estimate_noise_lundeby(
    energy: FloatArray,
    fs: float,
    smoothing_time_s: float,
) -> _NoiseResult:
    """Lundeby 噪声估计与 Schroeder 截断点搜索。

    对平方 RIR 做分块平均，先由尾部估计噪声功率，再用“噪声 + 10 dB”到
    “噪声 + 30 dB”的衰减段拟合直线并求与噪声底的交点；随后按衰减速度
    自适应选择块长，迭代更新噪声功率和交点。交点即 Schroeder 积分的
    有效终点。
    """

    n_samples = energy.size
    peak_energy = float(np.max(energy)) if n_samples > 0 else 0.0
    if peak_energy <= 0.0:
        return _NoiseResult(False, math.nan, math.nan, math.nan, math.nan, 0, False, "all_zero")

    floor = peak_energy * 1e-12
    tail = energy[int(0.9 * n_samples) :]
    if tail.size == 0:
        tail = energy[-1:]
    noise = max(float(np.mean(tail)), floor)
    if noise > 0.5 * peak_energy:
        return _NoiseResult(False, noise, math.nan, math.nan, math.nan, 0, False, "noise_too_high")

    block = round(smoothing_time_s * fs)
    block = max(1, min(block, max(1, n_samples // 8)))
    smoothed, time = _smooth_energy(energy, block, fs)
    if smoothed.size < 4:
        return _NoiseResult(False, noise, math.nan, math.nan, block / fs, 0, False, "rir_too_short")

    smoothed_db = 10.0 * np.log10(np.maximum(smoothed, _TINY))
    noise_db = 10.0 * math.log10(max(noise, _TINY))
    start = int(np.argmax(smoothed))
    above = np.flatnonzero(smoothed_db[start + 1 :] > noise_db + 10.0)
    if above.size == 0:
        return _NoiseResult(False, noise, math.nan, math.nan, block / fs, 0, False, "noise_dynamic_range")
    stop = start + 1 + int(above[-1])
    if stop - start < 2:
        return _NoiseResult(False, noise, math.nan, math.nan, block / fs, 0, False, "noise_dynamic_range")

    slope, intercept, _ = _fit_line(time[start : stop + 1], smoothed_db[start : stop + 1])
    if not np.isfinite(slope) or slope >= 0.0:
        return _NoiseResult(False, noise, math.nan, math.nan, block / fs, 0, False, "noise_regression_slope")

    dynamic_range = float(smoothed_db[stop] - smoothed_db[start])
    if dynamic_range > -5.0:
        return _NoiseResult(False, noise, math.nan, math.nan, block / fs, 0, False, "noise_dynamic_range")

    crossing = (noise_db - intercept) / slope
    blocks_in_decay = (-dynamic_range / 10.0) * 5.0
    new_block = round((time[stop] - time[start]) / blocks_in_decay * fs)
    new_block = max(max(4, round(1e-3 * fs)), min(new_block, max(1, n_samples // 8)))
    if abs(new_block - block) > max(1, block // 10):
        block = new_block
        smoothed, time = _smooth_energy(energy, block, fs)
        if smoothed.size < 8:
            return _NoiseResult(False, noise, math.nan, math.nan, block / fs, 0, False, "rir_too_short")
        smoothed_db = 10.0 * np.log10(np.maximum(smoothed, _TINY))

    converged = False
    iteration = 0
    for iteration in range(1, 31):
        block_time = block / fs
        idx_max = int(np.argmax(smoothed))
        idx_last = round(smoothed.size * 0.9)
        rel_decay = 10.0 / slope
        idx_below = int(max(1, round((crossing - rel_decay) / block_time)))
        noise_start = min(idx_last, idx_below)
        noise_start = min(max(noise_start, 0), smoothed.size - 1)
        new_noise = max(float(np.mean(smoothed[noise_start:])), floor)
        new_noise_db = 10.0 * math.log10(max(new_noise, _TINY))

        candidates = np.flatnonzero(smoothed_db[idx_max:] < new_noise_db + 30.0)
        if candidates.size == 0:
            return _NoiseResult(
                False, new_noise, math.nan, math.nan, block / fs, iteration, False, "noise_dynamic_range"
            )
        fit_start = idx_max + int(candidates[0])
        candidates = np.flatnonzero(smoothed_db[fit_start + 1 :] < new_noise_db + 10.0)
        if candidates.size == 0:
            return _NoiseResult(
                False, new_noise, math.nan, math.nan, block / fs, iteration, False, "noise_dynamic_range"
            )
        fit_stop = fit_start + 1 + int(candidates[0])
        if fit_stop - fit_start < 2:
            return _NoiseResult(
                False, new_noise, math.nan, math.nan, block / fs, iteration, False, "noise_dynamic_range"
            )

        slope, intercept, _ = _fit_line(
            time[fit_start : fit_stop + 1],
            smoothed_db[fit_start : fit_stop + 1],
        )
        if not np.isfinite(slope) or slope >= 0.0:
            return _NoiseResult(
                False, new_noise, math.nan, math.nan, block / fs, iteration, False, "noise_regression_slope"
            )

        previous = crossing
        crossing = (new_noise_db - intercept) / slope
        noise = new_noise
        if not np.isfinite(crossing) or crossing <= 0.0:
            return _NoiseResult(
                False, noise, math.nan, math.nan, block / fs, iteration, False, "noise_crossing_invalid"
            )
        if abs(previous - crossing) < 0.01:
            converged = True
            break

    if not converged:
        return _NoiseResult(False, noise, math.nan, math.nan, block / fs, iteration, False, "noise_not_converged")

    peak_snr_db = 10.0 * math.log10(peak_energy / noise)
    return _NoiseResult(
        True,
        noise,
        peak_snr_db,
        float(crossing),
        block / fs,
        iteration,
        True,
        None,
    )


def _compute_edc(energy: FloatArray) -> tuple[FloatArray, float]:
    """返回归一化 EDC 的 dB 曲线和总能量。

    低于总能量 ``1e-12``（-120 dB）的累积值用该下限截断，避免
    ``log10(0)`` 和极端负值；RT60 拟合区间远高于这个下限。
    """

    cumulative = np.cumsum(energy[::-1], dtype=np.float64)[::-1]
    total = float(cumulative[0]) if cumulative.size else 0.0
    if total <= 0.0:
        return np.empty(0, dtype=np.float64), 0.0
    floor = total * 1e-12
    edc_db = 10.0 * np.log10(np.maximum(cumulative, floor) / total)
    return edc_db, total


def _invalid_method(method: MethodName, reason: str) -> MethodEstimate:
    return MethodEstimate(
        method=method,
        rt60=math.nan,
        valid=False,
        reason=reason,
        r2=math.nan,
        slope_db_per_s=math.nan,
        fit_start_s=math.nan,
        fit_end_s=math.nan,
        fit_start_db=math.nan,
        fit_end_db=math.nan,
        fit_range_db=math.nan,
        n_samples=0,
        noise_contamination_db=math.nan,
        noise_margin_db=math.nan,
    )


def _fit_method(
    edc_db: FloatArray,
    fs: float,
    method: MethodName,
    noise_power: float,
    truncation_samples: int,
    total_energy: float,
    min_r2: float,
    min_noise_margin_db: float,
) -> MethodEstimate:
    upper_db, lower_db = _METHOD_RANGES[method]
    levels = edc_db

    if upper_db >= 0.0:
        start = 0
    else:
        candidates = np.flatnonzero(levels <= upper_db)
        if candidates.size == 0:
            return _invalid_method(method, "insufficient_decay")
        start = max(0, int(candidates[0]) - 1)

    candidates = np.flatnonzero(levels <= lower_db)
    if candidates.size == 0:
        return _invalid_method(method, "insufficient_decay")
    end = int(candidates[0])

    if end <= start or end - start + 1 < 8:
        return _invalid_method(method, "too_few_samples")

    time = np.arange(levels.size, dtype=np.float64) / fs
    fit_time = time[start : end + 1]
    fit_levels = levels[start : end + 1]
    slope, _, r2 = _fit_line(fit_time, fit_levels)
    if not np.isfinite(slope) or slope >= 0.0:
        return _invalid_method(method, "non_negative_slope")

    fit_range = float(fit_levels[-1] - fit_levels[0])
    required_range = lower_db - upper_db
    if fit_range > required_range + _FIT_RANGE_TOLERANCE_DB:
        return _invalid_method(method, "insufficient_decay")

    rt60 = -60.0 / slope
    if not np.isfinite(rt60) or rt60 <= 0.0:
        return _invalid_method(method, "invalid_slope")

    remaining_samples = max(truncation_samples - end, 0)
    noise_tail = noise_power * remaining_samples
    if noise_tail > 0.0 and total_energy > 0.0:
        contamination_db = 10.0 * math.log10(noise_tail / total_energy)
        noise_margin_db = float(fit_levels[-1] - contamination_db)
    else:
        contamination_db = -math.inf
        noise_margin_db = math.inf

    reason: str | None = None
    if r2 < min_r2:
        reason = "low_r2"
    elif noise_margin_db < min_noise_margin_db:
        reason = "noise_margin"

    return MethodEstimate(
        method=method,
        rt60=float(rt60),
        valid=reason is None,
        reason=reason,
        r2=float(r2),
        slope_db_per_s=float(slope),
        fit_start_s=float(fit_time[0]),
        fit_end_s=float(fit_time[-1]),
        fit_start_db=float(fit_levels[0]),
        fit_end_db=float(fit_levels[-1]),
        fit_range_db=fit_range,
        n_samples=int(fit_time.size),
        noise_contamination_db=float(contamination_db),
        noise_margin_db=float(noise_margin_db),
    )


def _invalid_estimate(
    reason: str,
    fs: float,
    onset_sample: int,
    n_samples: int,
    band_energy_db: float | None = None,
) -> RT60Estimate:
    methods = {name: _invalid_method(name, reason) for name in _METHODS}
    noise = NoiseInfo(
        algorithm="none",
        noise_power=math.nan,
        peak_snr_db=math.nan,
        noise_floor_db=math.nan,
        truncation_time_s=0.0,
        truncated_samples=0,
        truncated=False,
        smoothing_time_s=math.nan,
        iterations=0,
        converged=False,
        reason=reason,
    )
    return RT60Estimate(
        edt=math.nan,
        t20=math.nan,
        t30=math.nan,
        rt60=math.nan,
        method_used=None,
        r2=math.nan,
        valid=False,
        reason=reason,
        decay_range_db=math.nan,
        onset_sample=int(onset_sample),
        onset_time_s=onset_sample / fs,
        truncated_samples=0,
        band_energy_db=band_energy_db,
        methods=methods,
        noise=noise,
        edc_time_s=np.empty(0, dtype=np.float64),
        edc_db=np.empty(0, dtype=np.float64),
    )


def _minimum_samples(fs: float) -> int:
    return max(16, round(_MIN_DURATION_S * fs))


def _estimate_signal(
    x: FloatArray,
    fs: float,
    method: MethodOption,
    *,
    detect_onset: bool,
    min_r2: float,
    min_noise_margin_db: float,
    max_method_disagreement: float,
    smoothing_time_s: float | None,
    band_energy_db: float | None,
) -> RT60Estimate:
    min_samples = _minimum_samples(fs)
    if x.size < min_samples:
        return _invalid_estimate("rir_too_short", fs, 0, x.size, band_energy_db)
    if not np.any(x):
        return _invalid_estimate("all_zero", fs, 0, x.size, band_energy_db)

    onset = _detect_onset(x, fs) if detect_onset else 0
    y = np.asarray(x[onset:], dtype=np.float64)
    if y.size < min_samples:
        return _invalid_estimate("rir_too_short", fs, onset, y.size, band_energy_db)
    if not np.any(y):
        return _invalid_estimate("all_zero", fs, onset, y.size, band_energy_db)

    peak = float(np.max(np.abs(y)))
    if peak <= 0.0:
        return _invalid_estimate("all_zero", fs, onset, y.size, band_energy_db)
    # 静态峰值归一化只改变整体尺度，不改变 EDC 斜率；同时避免平方溢出。
    y = y / peak

    smoothing = 0.03 if smoothing_time_s is None else float(smoothing_time_s)
    energy = np.square(y, dtype=np.float64)
    noise_result = _estimate_noise_lundeby(energy, fs, smoothing)

    noise_power = noise_result.noise_power
    if np.isfinite(noise_result.crossing_s):
        truncation_samples = round(noise_result.crossing_s * fs)
        truncation_samples = min(max(truncation_samples, 0), y.size)
    else:
        truncation_samples = y.size
    noise_ok = noise_result.ok and truncation_samples >= min_samples
    if noise_result.ok and truncation_samples < min_samples:
        noise_result.ok = False
        noise_result.reason = "noise_crossing_too_early"

    if not noise_ok:
        truncation_samples = y.size

    edc_db, total_energy = _compute_edc(energy[:truncation_samples])
    if edc_db.size == 0 or total_energy <= 0.0:
        return _invalid_estimate("all_zero", fs, onset, y.size, band_energy_db)

    if np.isfinite(noise_power):
        noise_floor_db = 10.0 * math.log10(max(noise_power * truncation_samples, _TINY) / total_energy)
    else:
        noise_floor_db = math.nan

    methods = {
        name: _fit_method(
            edc_db,
            fs,
            name,
            noise_power,
            truncation_samples,
            total_energy,
            min_r2,
            min_noise_margin_db,
        )
        for name in _METHODS
    }

    if not noise_ok:
        problem = noise_result.reason or "unknown"
        methods = {
            name: replace(
                item,
                valid=False,
                reason=f"noise_estimation_failed: {problem}",
            )
            for name, item in methods.items()
        }

    chosen: MethodName | None = None
    disagreement_reason: str | None = None
    if noise_ok:
        if method == "auto":
            for name in ("T30", "T20"):
                if methods[name].valid:
                    chosen = name
                    break
            if chosen == "T20" and np.isfinite(methods["T30"].rt60):
                t20 = methods["T20"].rt60
                t30 = methods["T30"].rt60
                denominator = max(abs(t20), abs(t30), _TINY)
                disagreement = abs(t20 - t30) / denominator
                if disagreement > max_method_disagreement:
                    chosen = None
                    disagreement_reason = (
                        f"method_disagreement: T20={t20:.3f} s, T30={t30:.3f} s, "
                        f"difference={100.0 * disagreement:.1f}%"
                    )
        elif methods[method].valid:
            chosen = method

    if chosen is None:
        if not noise_ok:
            reason = f"noise_estimation_failed: {noise_result.reason or 'unknown'}"
        elif disagreement_reason is not None:
            reason = disagreement_reason
        elif method == "auto":
            parts = [f"{name}: {methods[name].reason}" for name in ("T30", "T20") if methods[name].reason]
            reason = "; ".join(parts) if parts else "no_reliable_method"
        else:
            reason = f"{method}: {methods[method].reason or 'not_valid'}"
    else:
        reason = None

    decay_range = max(
        0.0,
        min(-noise_floor_db, -float(edc_db[-1])) if np.isfinite(noise_floor_db) else -float(edc_db[-1]),
    )
    edc_time_s = np.arange(edc_db.size, dtype=np.float64) / fs
    noise_info = NoiseInfo(
        algorithm="lundeby",
        noise_power=float(noise_power) if np.isfinite(noise_power) else math.nan,
        peak_snr_db=float(noise_result.peak_snr_db) if np.isfinite(noise_result.peak_snr_db) else math.nan,
        noise_floor_db=float(noise_floor_db),
        truncation_time_s=truncation_samples / fs,
        truncated_samples=int(truncation_samples),
        truncated=bool(truncation_samples < y.size),
        smoothing_time_s=float(noise_result.block_s),
        iterations=int(noise_result.iterations),
        converged=bool(noise_result.converged),
        reason=noise_result.reason,
    )

    return RT60Estimate(
        edt=methods["EDT"].rt60,
        t20=methods["T20"].rt60,
        t30=methods["T30"].rt60,
        rt60=methods[chosen].rt60 if chosen else math.nan,
        method_used=chosen,
        r2=methods[chosen].r2 if chosen else math.nan,
        valid=chosen is not None,
        reason=reason,
        decay_range_db=float(decay_range),
        onset_sample=int(onset),
        onset_time_s=onset / fs,
        truncated_samples=int(truncation_samples),
        band_energy_db=band_energy_db,
        methods=methods,
        noise=noise_info,
        edc_time_s=edc_time_s,
        edc_db=edc_db,
    )


def _prepare_channels(rir: ArrayLike, time_axis: int) -> FloatArray:
    array = np.asarray(rir)
    if array.size == 0:
        raise ValueError("RIR 不能为空")
    if np.iscomplexobj(array):
        raise ValueError("RIR 必须为实数")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("RIR 必须是数值数组")
    data = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(data)):
        raise ValueError("RIR 中包含 NaN 或 Inf")

    if data.ndim == 1:
        if time_axis not in (0, -1):
            raise ValueError(f"一维 RIR 的 time_axis 只能为 0，实际为 {time_axis}")
        return data[:, None]
    if data.ndim != 2:
        raise ValueError(f"RIR 必须是 (N,) 或 (N, C)，实际形状为 {data.shape}")

    if time_axis in (0, -2):
        result = data
    elif time_axis in (1, -1):
        result = data.T
    else:
        raise ValueError(f"二维 RIR 的 time_axis 只能为 0 或 1，实际为 {time_axis}")
    if result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError(f"RIR 形状无效：{result.shape}")
    return result


def _resolve_bands(
    bands: Literal["octave", "none"] | Sequence[float] | None,
    fs: float,
) -> tuple[float, ...]:
    if bands is None:
        return ()
    if isinstance(bands, str):
        name = bands.strip().lower()
        if name in ("none", "off"):
            return ()
        if name != "octave":
            raise ValueError("bands 只支持 'octave'、'none' 或中心频率序列")
        requested = OCTAVE_BANDS_HZ
    else:
        requested = tuple(float(value) for value in bands)

    valid: list[float] = []
    for center in requested:
        if not np.isfinite(center) or center <= 0.0:
            raise ValueError(f"频带中心频率无效：{center!r}")
        upper = center * math.sqrt(2.0)
        if upper <= 0.95 * (fs / 2.0):
            valid.append(center)
    if requested and not valid:
        raise ValueError("没有频带满足当前采样率的 Nyquist 限制")
    return tuple(sorted(set(valid)))


def _filter_octave(x: FloatArray, fs: float, center: float) -> FloatArray:
    """零相位 SOS 倍频程滤波；两端补 50 ms 零以压低边缘瞬态。"""

    lower = center / math.sqrt(2.0)
    upper = center * math.sqrt(2.0)
    sos = signal.butter(4, [lower, upper], btype="bandpass", fs=fs, output="sos")
    pad = min(x.size // 2, max(round(0.05 * fs), 16))
    if pad <= 0:
        return np.asarray(signal.sosfiltfilt(sos, x), dtype=np.float64)
    padded = np.pad(x, (pad, pad), mode="constant")
    filtered = signal.sosfiltfilt(sos, padded, padtype=None)
    return np.asarray(filtered[pad:-pad], dtype=np.float64)


def estimate_rt60(
    rir: ArrayLike,
    fs: float = 16000,
    method: MethodOption = "auto",
    bands: Literal["octave", "none"] | Sequence[float] | None = None,
    *,
    time_axis: int = 0,
    detect_onset: bool = True,
    min_r2: float = 0.95,
    min_noise_margin_db: float = 10.0,
    max_method_disagreement: float = _MAX_METHOD_DISAGREEMENT,
) -> RT60Result:
    """从单通道或多通道 RIR 估计 RT60。

    Parameters
    ----------
    rir:
        ``(N,)`` 单通道或 ``(N, C)`` 多通道 RIR；默认时间轴为 0。
    fs:
        采样率，默认 16000 Hz。
    method:
        ``"auto"``、``"EDT"``、``"T20"`` 或 ``"T30"``。``auto`` 优先使用
        可信的 T30，其次 T20；不会用 EDT 顶替。
    bands:
        ``None`` 关闭分频带分析；``"octave"`` 使用 125--4000 Hz 六个倍频程；
        也可以传入自定义中心频率序列。
    time_axis:
        多通道输入的时间轴，0 或 1。单通道只能为 0。
    detect_onset:
        是否检测直达声起点。默认开启。
    min_r2:
        线性拟合可信度阈值，默认 0.95。
    min_noise_margin_db:
        拟合终点与噪声底之间要求的最小余量，默认 10 dB。
    max_method_disagreement:
        ``auto`` 回退到 T20 时，允许的 T20/T30 相对差异；超过该值时判为
        无效，默认 0.25（25%）。

    Returns
    -------
    RT60Result
        每个通道包含 EDT、T20、T30、最终 RT60、方法、R²、有效性、原因、
        噪声诊断和 EDC 数据；多通道结果附带剔除无效通道后的统计。
    """

    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError(f"fs 必须是有限正数，实际为 {fs!r}")
    if not 0.0 <= min_r2 <= 1.0:
        raise ValueError("min_r2 必须在 [0, 1] 内")
    if min_noise_margin_db < 0.0:
        raise ValueError("min_noise_margin_db 不能为负")
    if max_method_disagreement < 0.0:
        raise ValueError("max_method_disagreement 不能为负")

    method_name = method.strip()
    if method_name.lower() == "auto":
        method_normalized: MethodOption = "auto"
    else:
        method_normalized = method_name.upper()  # type: ignore[assignment]
    if method_normalized not in ("auto", *_METHODS):
        raise ValueError(f"method 必须是 auto、EDT、T20 或 T30，实际为 {method!r}")

    data = _prepare_channels(rir, time_axis)
    centers = _resolve_bands(bands, float(fs))

    channels: list[ChannelRT60Result] = []
    for channel_index in range(data.shape[1]):
        signal_data = data[:, channel_index]
        broadband = _estimate_signal(
            signal_data,
            float(fs),
            method_normalized,
            detect_onset=detect_onset,
            min_r2=min_r2,
            min_noise_margin_db=min_noise_margin_db,
            max_method_disagreement=max_method_disagreement,
            smoothing_time_s=None,
            band_energy_db=None,
        )

        band_results: dict[float, RT60Estimate] = {}
        if centers:
            if not np.any(signal_data):
                for center in centers:
                    band_results[center] = _invalid_estimate(
                        "all_zero",
                        float(fs),
                        0,
                        signal_data.size,
                        None,
                    )
            elif signal_data.size < _minimum_samples(float(fs)):
                for center in centers:
                    band_results[center] = _invalid_estimate(
                        "rir_too_short",
                        float(fs),
                        0,
                        signal_data.size,
                        None,
                    )
            else:
                broadband_peak = float(np.max(np.abs(signal_data)))
                for center in centers:
                    filtered = _filter_octave(signal_data, float(fs), center)
                    band_peak = float(np.max(np.abs(filtered))) if filtered.size else 0.0
                    if broadband_peak > 0.0 and band_peak > 0.0:
                        band_energy_db = 20.0 * math.log10(band_peak / broadband_peak)
                    else:
                        band_energy_db = -math.inf
                    if band_energy_db < _BAND_ENERGY_FLOOR_DB:
                        band_results[center] = _invalid_estimate(
                            "band_energy_too_low",
                            float(fs),
                            0,
                            filtered.size,
                            band_energy_db,
                        )
                        continue
                    band_results[center] = _estimate_signal(
                        filtered,
                        float(fs),
                        method_normalized,
                        detect_onset=detect_onset,
                        min_r2=min_r2,
                        min_noise_margin_db=min_noise_margin_db,
                        max_method_disagreement=max_method_disagreement,
                        smoothing_time_s=0.8 / center + 0.01,
                        band_energy_db=band_energy_db,
                    )

        channels.append(ChannelRT60Result(channel_index, broadband, band_results))

    band_output: dict[float, BandRT60Result] = {}
    for center in centers:
        values = [channel.bands[center] for channel in channels]
        by_channel_stats = _compute_stats(
            [item.rt60 for item in values if item.valid],
            len(values),
        )
        band_output[center] = BandRT60Result(center, tuple(values), by_channel_stats)

    channel_stats = _compute_stats(
        [channel.rt60 for channel in channels],
        len(channels),
    )
    method_stats = {
        name: _compute_stats(
            [
                channel.broadband.methods[name].rt60
                for channel in channels
                if channel.broadband.methods[name].valid
            ],
            len(channels),
        )
        for name in _METHODS
    }

    return RT60Result(
        fs=float(fs),
        method=method_normalized,
        channels=tuple(channels),
        bands_hz=centers,
        bands=band_output,
        stats=channel_stats,
        method_stats=method_stats,
        rir=data,
    )
