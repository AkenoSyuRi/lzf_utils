"""MATLAB Audio Toolbox ``detectSpeech`` 的 Python 复刻版。

本文件根据用户提供的以下 MATLAB 源码逐项翻译：

- ``detectSpeech.m``
- ``audio.internal.buffer`` 对应的 ``buffer.m``
- ``spectralSpread.m``

核心行为：

1. 核心函数默认使用 30 ms periodic Hann 窗且无重叠；命令行默认使用半窗重叠；
2. 仅保留完整帧，不给末帧补零；
3. 计算带窗短时能量和 magnitude-spectrum spectral spread；
4. 两个特征分别连续通过两次长度为 5 的 moving median；
5. 用特征直方图局部峰值自动估计阈值；
6. 能量和频谱扩展度同时超过阈值时，帧才被判为语音；
7. 重叠帧按原 MATLAB 辅助函数投票，再按 MergeDistance 合并区间。

直接运行：

    uv run python -m lzf_utils.audio.detect_speech input.wav
    uv run python -m lzf_utils.audio.detect_speech input.wav --out-wav result.wav --csv result.csv

命令行默认使用半窗重叠，以提高起音和尾音的边界分辨率；传入
``--overlap-ms 0`` 可恢复 MATLAB ``detectSpeech`` 的默认无重叠行为。

作为模块使用：

    from lzf_utils.audio import detect_speech
    intervals, thresholds = detect_speech(audio, fs)

返回 0-based、右端不包含的区间，即 ``audio[start:stop]``。

说明：MATLAB 内部 ``audio.internal.spectralDescriptors.stft`` 和
``histcounts(feature, numBins)`` 的实现未包含在用户提供的源码中。本版使用
SciPy FFT 和 NumPy 等宽直方图对应；绝大多数音频上结果应高度接近，但自动阈值
可能存在小数值差异。可传入 MATLAB 已计算的 ``thresholds`` 消除直方图差异。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import fft as scipy_fft
from scipy.io import wavfile

FloatArray = NDArray[np.float32] | NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class DetectSpeechDebug:
    """用于对照 MATLAB 中间结果的调试数据。"""

    frame_length: int
    hop_length: int
    frame_energy: NDArray[np.floating]
    smoothed_energy: NDArray[np.floating]
    spectral_spread: NDArray[np.floating]
    smoothed_spectral_spread: NDArray[np.floating]
    speech_frame_mask: NDArray[np.bool_]
    speech_sample_mask: NDArray[np.bool_]
    intervals: IntArray


def _matlab_round_positive(value: float) -> int:
    """匹配本算法中正数输入的 MATLAB round（0.5 向远离 0 的方向）。"""

    if not math.isfinite(value) or value < 0:
        raise ValueError(f"需要有限非负值，实际为 {value!r}")
    return int(math.floor(value + 0.5))


def periodic_hann(length: int, dtype: np.dtype | type = np.float64) -> NDArray:
    """等价于 ``hann(length, 'periodic')``。"""

    if length < 1:
        raise ValueError("Hann 窗长度必须大于 0")
    n = np.arange(length, dtype=np.float64)
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / float(length))
    return window.astype(dtype, copy=False)


def _as_mono_float(audio_in: ArrayLike) -> FloatArray:
    """将输入转换成有限的单声道 float32/float64 向量。"""

    x = np.asarray(audio_in)
    if x.ndim == 2 and 1 in x.shape:
        x = x.reshape(-1)
    if x.ndim != 1:
        raise ValueError(f"audio_in 必须是单声道一维向量或 Nx1 列向量，实际形状为 {x.shape}")
    if x.size == 0:
        raise ValueError("audio_in 不能为空")
    if np.iscomplexobj(x):
        raise ValueError("audio_in 必须为实数")

    if x.dtype == np.float32:
        y = np.asarray(x, dtype=np.float32)
    else:
        y = np.asarray(x, dtype=np.float64)

    if not np.all(np.isfinite(y)):
        raise ValueError("audio_in 中包含 NaN 或 Inf")
    return y


def _buffer_complete_frames(x: FloatArray, window_length: int, hop_length: int) -> NDArray[np.floating]:
    """翻译 ``buffer.m``：只提取完整帧，不给末帧补零。

    返回形状为 ``(num_frames, window_length)``，与 MATLAB 源码中的
    ``(window_length, num_frames)`` 互为转置。
    """

    num_hops = (x.size - window_length) // hop_length + 1
    if num_hops <= 0:
        raise ValueError("音频长度必须不小于窗长")

    # sliding_window_view 返回视图，不复制整段分帧数据。
    all_frames = np.lib.stride_tricks.sliding_window_view(x, window_length)
    return all_frames[0 : num_hops * hop_length : hop_length]


def _moving_median_shrink(values: NDArray[np.floating], length: int) -> NDArray[np.floating]:
    """对应 MATLAB ``movmedian(values, length)`` 的默认 shrink 端点规则。"""

    x = np.asarray(values)
    if x.ndim != 1:
        raise ValueError("moving median 输入必须是一维")
    if length <= 0 or length % 2 == 0:
        raise ValueError("本实现要求 moving median 长度为正奇数")

    radius = length // 2
    out = np.empty_like(x)
    for i in range(x.size):
        start = max(0, i - radius)
        stop = min(x.size, i + radius + 1)
        out[i] = np.median(x[start:stop])
    return out


def spectral_spread_from_magnitude(
    magnitude_spectrum: NDArray[np.floating],
    frequency_vector: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """翻译频域输入形式的 ``spectralSpread.m``。

    参数
    ----
    magnitude_spectrum:
        ``(num_frames, num_frequency_bins)``。
    frequency_vector:
        与频率维对应的 Hz 向量。

    返回
    ----
    spread, centroid:
        每帧的频谱扩展度与频谱质心，单位为 Hz。
    """

    spectrum = np.asarray(magnitude_spectrum)
    frequencies = np.asarray(frequency_vector, dtype=spectrum.dtype)
    if spectrum.ndim != 2:
        raise ValueError("magnitude_spectrum 必须是二维数组")
    if frequencies.ndim != 1 or frequencies.size != spectrum.shape[1]:
        raise ValueError("frequency_vector 长度必须等于频率 bin 数")

    denominator = np.sum(spectrum, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        centroid = np.sum(spectrum * frequencies[None, :], axis=1) / denominator
        centered = frequencies[None, :] - centroid[:, None]
        spread = np.sqrt(np.sum((centered * centered) * spectrum, axis=1) / denominator)
    return spread, centroid


def _f_peaks(histogram_counts: NDArray[np.integer]) -> IntArray:
    """翻译 MATLAB ``fPeaks``。

    某 bin 的计数必须严格大于其前后各 3 个 bin；越界邻居视作 0。
    MATLAB 源码还会强制最后一个 bin 不能成为峰值。
    """

    counts = np.asarray(histogram_counts)
    if counts.ndim != 1:
        raise ValueError("histogram_counts 必须是一维")
    if counts.size == 0:
        return np.empty(0, dtype=np.int64)

    peaks: list[int] = []
    for i in range(counts.size):
        center = 0 if i == counts.size - 1 else counts[i]
        is_peak = True
        for offset in (-3, -2, -1, 1, 2, 3):
            j = i + offset
            neighbor = counts[j] if 0 <= j < counts.size else 0
            if not center > neighbor:
                is_peak = False
                break
        if is_peak:
            peaks.append(i)
    return np.asarray(peaks, dtype=np.int64)


def _histcounts_uniform(feature: NDArray[np.floating], num_bins: int) -> tuple[IntArray, NDArray[np.floating]]:
    """NumPy 对应 ``histcounts(feature, num_bins)``。

    MATLAB 对自动 bin limits 有私有数值细节；这里使用覆盖数据范围的等宽 bin。
    ``numpy.histogram`` 与 MATLAB 一样，最后一个 bin 包含右边界。
    """

    finite = np.asarray(feature)[np.isfinite(feature)]
    if finite.size == 0:
        raise ValueError("特征中没有有限数值")
    counts, edges = np.histogram(finite, bins=num_bins)
    return counts.astype(np.int64, copy=False), edges.astype(feature.dtype, copy=False)


def _get_thresholds_from_feature(
    feature: NDArray[np.floating], bins: int, feature_type: Literal["specspread", "energy"]
) -> tuple[np.floating, np.floating]:
    """翻译 MATLAB ``getThreshsFromFeature``，返回 ``M1, M2``。"""

    x = np.asarray(feature)
    hist_bins = max(10, _matlab_round_positive(x.size / float(bins)))
    mean_feature = np.mean(x)
    counts, edges = _histcounts_uniform(x, hist_bins)

    if feature_type == "specspread":
        # MATLAB 使用严格的 edgesFeature(1) == 0。
        if edges.size > 1 and edges[0] == 0:
            counts = counts[1:]
            edges = edges[1:]
        min_value = mean_feature / x.dtype.type(2.0)
    elif feature_type == "energy":
        min_value = np.min(x)
    else:
        raise ValueError(f"未知 feature_type: {feature_type}")

    peaks = _f_peaks(counts)

    if feature_type == "energy" and peaks.size >= 2:
        # 逐字对应源码中的：
        # all(max(nFeature(peaksIdx(1:2))) > mFeature)
        if np.max(counts[peaks[:2]]) > mean_feature:
            if edges[peaks[1]] > mean_feature:
                peaks = peaks[:1]
            elif edges[peaks[0]] > mean_feature:
                peaks = peaks[1:]

    if peaks.size == 0:
        m1 = mean_feature / x.dtype.type(2.0)
        m2 = min_value
    else:
        centers = (edges[:-1] + edges[1:]) / x.dtype.type(2.0)
        if peaks.size == 1:
            m1 = centers[peaks[0]]
            m2 = min_value
        else:
            m2 = centers[peaks[0]]
            m1 = centers[peaks[1]]

    return m1, m2


def _debuffer_frame_overlap(
    speech_mask: NDArray[np.bool_], frame_length: int, overlap_length: int
) -> tuple[NDArray[np.bool_], int]:
    """逐项翻译 MATLAB ``debufferFrameOverlap``。"""

    hop_length = frame_length - overlap_length
    num_shared_frames = frame_length // hop_length

    extended = np.concatenate(
        [
            speech_mask.astype(np.float64, copy=False),
            np.zeros(num_shared_frames - 1, dtype=np.float64),
        ]
    )

    # 等价于 filter(ones(1,numSharedFrames), 1, extended)。
    full_convolution = np.convolve(extended, np.ones(num_shared_frames, dtype=np.float64), mode="full")
    nearest_votes = full_convolution[: extended.size]

    begin_threshold = np.arange(1, num_shared_frames, dtype=np.float64) / 2.0
    end_threshold = begin_threshold[::-1]
    middle_length = nearest_votes.size - 2 * (num_shared_frames - 1)
    if middle_length < 0:
        raise ValueError(
            "重叠比例与音频长度使 MATLAB debufferFrameOverlap 的阈值长度为负；" "请减小 OverlapLength 或使用更长音频"
        )
    middle_threshold = np.ones(middle_length, dtype=np.float64)
    threshold = np.concatenate([begin_threshold, middle_threshold, end_threshold])
    return nearest_votes >= threshold, hop_length


def _sample_mask_to_intervals(
    sample_mask: NDArray[np.bool_],
    first_debuffered_value: bool,
    merge_distance: int,
) -> IntArray:
    """从样本 mask 生成 0-based、右端不包含的 ``[start, stop)`` 区间。"""

    difference = np.diff(np.concatenate([sample_mask.astype(np.int8), np.zeros(1, dtype=np.int8)]))

    stop_indices = np.flatnonzero(difference == -1).astype(np.int64) + 1
    rising = np.flatnonzero(difference == 1).astype(np.int64)
    if first_debuffered_value:
        start_indices = np.concatenate([np.array([0], dtype=np.int64), rising])
    else:
        start_indices = rising

    if start_indices.size == 0 or stop_indices.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if start_indices.size != stop_indices.size:
        raise RuntimeError("内部区间边界数量不一致：" f"start={start_indices.size}, stop={stop_indices.size}")

    if start_indices.size > 1:
        # next_start - prev_stop + 1 对应原算法对闭区间端点差的合并判断。
        merge_mask = start_indices[1:] - stop_indices[: start_indices.size - 1] + 1 <= merge_distance
    else:
        merge_mask = np.empty(0, dtype=bool)

    keep_boundary = ~merge_mask
    selected_starts = np.concatenate([start_indices[:1], start_indices[1:][keep_boundary]])
    selected_stops = np.concatenate([stop_indices[:-1][keep_boundary], stop_indices[-1:]])
    return np.column_stack([selected_starts, selected_stops]).astype(np.int64, copy=False)


def detect_speech(
    audio_in: ArrayLike,
    fs: float,
    *,
    window: ArrayLike | None = None,
    overlap_length: int = 0,
    merge_distance: int | None = None,
    thresholds: Sequence[float] | NDArray[np.floating] | None = None,
    return_debug: bool = False,
) -> tuple[IntArray, NDArray[np.floating]] | tuple[IntArray, NDArray[np.floating], DetectSpeechDebug]:
    """检测单声道音频中的语音区间。

    参数与 MATLAB ``detectSpeech`` 对应。默认 ``window`` 为 30 ms periodic
    Hann，``overlap_length`` 为 0，``merge_distance`` 为 5 倍窗长。

    返回 0-based、右端不包含的 ``[start, stop)``，可直接用于 ``audio[start:stop]``。
    """

    x = _as_mono_float(audio_in)

    if not math.isfinite(float(fs)) or float(fs) <= 20:
        raise ValueError("fs 必须是大于 20 Hz 的有限正数")
    fs_float = float(fs)

    if window is None:
        frame_length_default = _matlab_round_positive(0.03 * fs_float)
        analysis_window = periodic_hann(frame_length_default, dtype=x.dtype)
    else:
        analysis_window = np.asarray(window, dtype=x.dtype).reshape(-1)
        if analysis_window.size == 0 or not np.all(np.isfinite(analysis_window)):
            raise ValueError("window 必须是非空有限实数向量")

    frame_length = int(analysis_window.size)
    if frame_length < 2 or frame_length > x.size:
        raise ValueError(f"window 长度必须位于 [2, {x.size}]，实际为 {frame_length}")
    if isinstance(overlap_length, bool) or int(overlap_length) != overlap_length:
        raise ValueError("overlap_length 必须是整数")
    overlap_length = int(overlap_length)
    if overlap_length < 0 or overlap_length >= frame_length:
        raise ValueError(f"overlap_length 必须位于 [0, {frame_length - 1}]")

    if merge_distance is None:
        merge_distance = frame_length * 5
    if isinstance(merge_distance, bool) or int(merge_distance) != merge_distance:
        raise ValueError("merge_distance 必须是整数")
    merge_distance = int(merge_distance)
    if merge_distance < 0:
        raise ValueError("merge_distance 不能为负数")

    threshold_array: NDArray[np.floating] | None
    if thresholds is None:
        threshold_array = None
    else:
        threshold_array = np.asarray(thresholds, dtype=x.dtype).reshape(-1)
        if threshold_array.size != 2 or not np.all(np.isfinite(threshold_array)) or np.any(threshold_array < 0):
            raise ValueError("thresholds 必须是两个有限非负数：[energy, spread]")

    # Step 1: normalize and extract complete frames.
    signal_max = np.max(np.abs(x))
    normalized = x / signal_max if signal_max > 0 else x.copy()
    hop_length = frame_length - overlap_length
    frames = _buffer_complete_frames(normalized, frame_length, hop_length)

    # MATLAB: energy = (window.').^2 * frames.^2
    frame_energy = np.sum(
        (frames * analysis_window[None, :]) ** 2,
        axis=1,
        dtype=x.dtype,
    )
    smoothed_energy = _moving_median_shrink(_moving_median_shrink(frame_energy, 5), 5)

    # MATLAB internal STFT: one spectrum per already-buffered frame,
    # FFTLength=2*frameLength, Range=[0, floor(fs/2)], SpectrumType='magnitude'.
    fft_length = 2 * frame_length
    spectrum = scipy_fft.rfft(frames * analysis_window[None, :], n=fft_length, axis=1)
    magnitude = np.abs(spectrum)
    frequencies = scipy_fft.rfftfreq(fft_length, d=1.0 / fs_float)
    frequency_mask = frequencies <= math.floor(fs_float / 2.0)
    magnitude = magnitude[:, frequency_mask]
    frequencies = frequencies[frequency_mask].astype(magnitude.dtype, copy=False)

    spectral_spread, _ = spectral_spread_from_magnitude(magnitude, frequencies)
    spectral_spread = spectral_spread / x.dtype.type(fs_float / 2.0)
    spectral_spread = np.asarray(spectral_spread, dtype=x.dtype)
    spectral_spread[frame_energy < x.dtype.type(0.05)] = x.dtype.type(0.0)
    smoothed_spread = _moving_median_shrink(_moving_median_shrink(spectral_spread, 5), 5)

    # Step 2: determine thresholds.
    if threshold_array is None:
        spread_m1, spread_m2 = _get_thresholds_from_feature(smoothed_spread, 15, "specspread")
        energy_m1, energy_m2 = _get_thresholds_from_feature(smoothed_energy, 15, "energy")
        weight = x.dtype.type(1.0 / 6.0)
        spread_threshold = weight * (x.dtype.type(5.0) * spread_m2 + spread_m1) * x.dtype.type(0.8)
        energy_threshold = weight * (x.dtype.type(5.0) * energy_m2 + energy_m1)
        used_thresholds = np.asarray([energy_threshold, spread_threshold], dtype=x.dtype)
    else:
        used_thresholds = threshold_array.copy()
        energy_threshold = used_thresholds[0]
        spread_threshold = used_thresholds[1]

    # Step 3: both criteria must be true.
    speech_frame_mask = (smoothed_spread > spread_threshold) & (smoothed_energy > energy_threshold)

    # Step 4: overlap vote, frame-to-sample expansion, and region merge.
    if overlap_length > 0:
        debuffered, output_block_length = _debuffer_frame_overlap(speech_frame_mask, frame_length, overlap_length)
    else:
        debuffered = speech_frame_mask
        output_block_length = frame_length

    represented_sample_count = debuffered.size * output_block_length
    if represented_sample_count > x.size:
        raise RuntimeError(
            "按 MATLAB debuffer 规则展开后的样本数超过输入长度：" f"{represented_sample_count} > {x.size}"
        )
    speech_sample_mask = np.concatenate(
        [
            np.repeat(debuffered, output_block_length),
            np.zeros(x.size - represented_sample_count, dtype=bool),
        ]
    )

    intervals = _sample_mask_to_intervals(
        speech_sample_mask,
        first_debuffered_value=bool(debuffered[0]),
        merge_distance=merge_distance,
    )

    if not return_debug:
        return intervals, used_thresholds

    debug = DetectSpeechDebug(
        frame_length=frame_length,
        hop_length=hop_length,
        frame_energy=frame_energy,
        smoothed_energy=smoothed_energy,
        spectral_spread=spectral_spread,
        smoothed_spectral_spread=smoothed_spread,
        speech_frame_mask=speech_frame_mask,
        speech_sample_mask=speech_sample_mask,
        intervals=intervals,
    )
    return intervals, used_thresholds, debug


def _pcm_to_float(data: NDArray) -> NDArray[np.float64]:
    """模拟 ``audioread`` 的常见 PCM 归一化行为。"""

    if np.issubdtype(data.dtype, np.floating):
        return np.asarray(data, dtype=np.float64)
    if data.dtype == np.uint8:
        return (data.astype(np.float64) - 128.0) / 128.0
    if np.issubdtype(data.dtype, np.signedinteger):
        scale = float(-np.iinfo(data.dtype).min)
        return data.astype(np.float64) / scale
    raise TypeError(f"不支持的 WAV 数据类型：{data.dtype}")


def read_wav_mono(
    path: str | Path,
    *,
    channel: int | None = None,
    mixdown: bool = False,
) -> tuple[NDArray[np.float64], int]:
    """读取 WAV；多通道文件必须指定 channel 或 mixdown。"""

    sample_rate, raw = wavfile.read(Path(path))
    audio = _pcm_to_float(np.asarray(raw))

    if audio.ndim == 1:
        if channel not in (None, 0):
            raise ValueError("单声道 WAV 只能使用 channel=0")
        return audio, int(sample_rate)

    if audio.ndim != 2:
        raise ValueError(f"只支持一维或二维 WAV 数据，实际形状为 {audio.shape}")
    if channel is not None and mixdown:
        raise ValueError("--channel 与 --mixdown 不能同时使用")
    if mixdown:
        return np.mean(audio, axis=1), int(sample_rate)
    if channel is None:
        raise ValueError(f"输入 WAV 有 {audio.shape[1]} 个通道；请指定 --channel，或使用 --mixdown")
    if channel < 0 or channel >= audio.shape[1]:
        raise ValueError(f"channel 必须位于 [0, {audio.shape[1] - 1}]，实际为 {channel}")
    return audio[:, channel], int(sample_rate)


def _float_to_int16(audio: NDArray[np.floating]) -> NDArray[np.int16]:
    """将有符号 PCM 的 float 归一化结果写回 int16。"""

    scale = float(-np.iinfo(np.int16).min)
    return np.clip(np.rint(np.asarray(audio) * scale), -32768, 32767).astype(np.int16)


def _write_stereo_vad_wav(
    path: Path,
    audio: NDArray[np.floating],
    sample_rate: int,
    intervals: IntArray,
) -> None:
    """保存双通道 WAV：L 为原音频，R 为 VAD 方波（峰值 10000）。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    left = _float_to_int16(audio)
    right = np.zeros(left.size, dtype=np.int16)
    for start, stop in intervals:
        right[int(start) : int(stop)] = 10000
    wavfile.write(path, int(sample_rate), np.column_stack((left, right)))


def _write_csv(
    path: Path,
    intervals: IntArray,
    sample_rate: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "segment",
                "start",
                "stop",
                "start_seconds",
                "stop_seconds",
                "duration_seconds",
            ]
        )
        for number, (start, stop) in enumerate(intervals, start=1):
            writer.writerow(
                [
                    number,
                    int(start),
                    int(stop),
                    start / sample_rate,
                    stop / sample_rate,
                    (stop - start) / sample_rate,
                ]
            )


def _write_json(
    path: Path,
    input_path: Path,
    sample_rate: int,
    thresholds: NDArray[np.floating],
    intervals: IntArray,
    debug: DetectSpeechDebug,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input": str(input_path),
        "sample_rate": sample_rate,
        "frame_length": debug.frame_length,
        "hop_length": debug.hop_length,
        "thresholds": {
            "energy": float(thresholds[0]),
            "spectral_spread": float(thresholds[1]),
        },
        "interval_convention": "0-based [start, stop), stop exclusive",
        "segments": [
            {
                "start": int(start),
                "stop": int(stop),
                "start_seconds": start / sample_rate,
                "stop_seconds": stop / sample_rate,
                "duration_seconds": (stop - start) / sample_rate,
            }
            for start, stop in intervals
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _plot_detection(
    audio: NDArray[np.float64],
    sample_rate: int,
    intervals: IntArray,
    output: str,
) -> None:
    import matplotlib.pyplot as plt

    time = np.arange(audio.size, dtype=np.float64) / sample_rate
    figure, axis = plt.subplots(figsize=(14, 4.8))
    axis.plot(time, audio, linewidth=0.7, label="audio")
    for index, (start, stop) in enumerate(intervals):
        label = "detected speech" if index == 0 else None
        axis.axvspan(start / sample_rate, stop / sample_rate, alpha=0.2, label=label)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Amplitude")
    axis.set_title("detectSpeech Python translation")
    axis.grid(True, alpha=0.3)
    if intervals.size:
        axis.legend(loc="upper right")
    axis.set_xlim(0.0, audio.size / sample_rate)
    figure.tight_layout()

    if output == "-":
        plt.show()
    else:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MATLAB Audio Toolbox detectSpeech 的 Python 复刻版")
    parser.add_argument("wav", type=Path, help="待检测的 WAV 文件")
    parser.add_argument(
        "--window-ms",
        type=float,
        default=30.0,
        help="periodic Hann 窗长，单位 ms；默认 30",
    )
    parser.add_argument(
        "--overlap-ms",
        type=float,
        default=None,
        help="相邻窗重叠，单位 ms；默认使用窗长的一半，传 0 可复现 MATLAB 默认值",
    )
    parser.add_argument(
        "--merge-ms",
        type=float,
        default=None,
        help="合并间隔，单位 ms；默认 5 倍窗长",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs=2,
        metavar=("ENERGY", "SPREAD"),
        help="复用固定阈值，顺序为 energy、spectral spread",
    )
    parser.add_argument("--channel", type=int, help="多通道 WAV 使用的通道号，0-based")
    parser.add_argument("--mixdown", action="store_true", help="多通道 WAV 先平均为单声道")
    parser.add_argument(
        "--out-wav",
        type=Path,
        help="保存双通道 WAV：L 为原音频，R 为 VAD 结果（峰值 10000）",
    )
    parser.add_argument("--csv", type=Path, help="保存检测区间 CSV")
    parser.add_argument("--json", type=Path, help="保存检测结果 JSON")
    parser.add_argument(
        "--plot",
        nargs="?",
        const="-",
        metavar="PNG",
        help="不带路径时显示图；带路径时保存 PNG",
    )
    parser.add_argument("--debug-npz", type=Path, help="保存中间特征、帧 mask 和样本 mask")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    try:
        audio, sample_rate = read_wav_mono(args.wav, channel=args.channel, mixdown=args.mixdown)
        window_length = _matlab_round_positive(args.window_ms * sample_rate / 1000.0)
        overlap_length = (
            _matlab_round_positive(window_length * 0.5)
            if args.overlap_ms is None
            else _matlab_round_positive(args.overlap_ms * sample_rate / 1000.0)
        )
        merge_distance = None if args.merge_ms is None else _matlab_round_positive(args.merge_ms * sample_rate / 1000.0)
        window = periodic_hann(window_length, dtype=np.float64)

        intervals, used_thresholds, debug = detect_speech(
            audio,
            sample_rate,
            window=window,
            overlap_length=overlap_length,
            merge_distance=merge_distance,
            thresholds=args.thresholds,
            return_debug=True,
        )
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))

    print(f"input: {args.wav}")
    print(f"sample_rate: {sample_rate} Hz")
    print(f"frame_length: {debug.frame_length} samples " f"({debug.frame_length / sample_rate * 1000.0:.3f} ms)")
    print(f"hop_length: {debug.hop_length} samples " f"({debug.hop_length / sample_rate * 1000.0:.3f} ms)")
    print(
        "thresholds: " f"energy={float(used_thresholds[0]):.10g}, " f"spectral_spread={float(used_thresholds[1]):.10g}"
    )
    print(f"segments: {intervals.shape[0]}")
    for index, (start, stop) in enumerate(intervals, start=1):
        print(
            f"  {index:03d}: "
            f"[{start}, {stop})  "
            f"time=[{start / sample_rate:.6f}, {stop / sample_rate:.6f}) s  "
            f"duration={(stop - start) / sample_rate:.6f} s"
        )

    if args.out_wav is not None:
        _write_stereo_vad_wav(args.out_wav, audio, sample_rate, intervals)
        print(f"out_wav: {args.out_wav}")
    if args.csv is not None:
        _write_csv(args.csv, intervals, sample_rate)
        print(f"csv: {args.csv}")
    if args.json is not None:
        _write_json(
            args.json,
            args.wav,
            sample_rate,
            used_thresholds,
            intervals,
            debug,
        )
        print(f"json: {args.json}")
    if args.debug_npz is not None:
        args.debug_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.debug_npz,
            frame_length=np.int64(debug.frame_length),
            hop_length=np.int64(debug.hop_length),
            thresholds=used_thresholds,
            frame_energy=debug.frame_energy,
            smoothed_energy=debug.smoothed_energy,
            spectral_spread=debug.spectral_spread,
            smoothed_spectral_spread=debug.smoothed_spectral_spread,
            speech_frame_mask=debug.speech_frame_mask,
            speech_sample_mask=debug.speech_sample_mask,
            intervals=intervals,
        )
        print(f"debug_npz: {args.debug_npz}")
    if args.plot is not None:
        _plot_detection(audio, sample_rate, intervals, args.plot)
        if args.plot != "-":
            print(f"plot: {args.plot}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
