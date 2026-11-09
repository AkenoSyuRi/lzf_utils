"""扫描多通道录音的均方 dBFS 上下界。

统一重采样到 16 kHz，默认用 4 秒窗、1 秒步长。每组通道用全窗均方计算 dBFS，
满幅是 ±1.0。MIC 按各文件的底噪估计加余量筛窗。REF 排除低于静音门限
或仅在窗头、窗尾短暂有活动的窗。默认对 REF 静默窗额外筛掉活动不足窗长
50% 的 MIC 窗，可用 ``--no-mic-spectral-vad`` 关闭。最低窗和最高窗同时输出
在原文件中的起止时间。

命令行::

    uv run scan_dbfs --mic-indices 5,6,7,8 --ref-indices 9 /path/to/dir /path/to/file.wav
    uv run python -m lzf_utils.audio.scan_dbfs --mic-indices 5,6,7,8 --ref-indices 9 input.wav

作为模块使用::

    from lzf_utils.audio import scan_dbfs_file

    mic_bounds, ref_bounds = scan_dbfs_file(
        "capture.wav",
        mic_indices=[5, 6, 7, 8],
        ref_indices=[9],
        window_sec=4.0,
    )
    print(mic_bounds.low, mic_bounds.high)

``Bounds.low`` / ``Bounds.high`` 的单位是 dBFS，``None`` 表示没有有效窗。
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

# 与训练样本的采样率和时长对齐。
SAMPLE_RATE = 16000
DEFAULT_WINDOW_SEC = 4.0
DEFAULT_HOP_SEC = 1.0
# 低于该门限视为静音。房间底噪通常高于此值，数字静音会被排除。
DEFAULT_MIN_DBFS = -90.0
DEFAULT_MIC_NOISE_MARGIN_DB = 6.0
DBFS_POWER_EPS = 1e-20
FRAME_SAMPLES = SAMPLE_RATE // 50  # 20 ms
MIC_NOISE_PERCENTILE = 10
MIN_NOISE_FRAMES = SAMPLE_RATE // FRAME_SAMPLES  # 至少约 1 秒的底噪候选小段
MIC_FLUX_SMOOTH_FRAMES = 7  # 140 ms
MIC_FLUX_LAG_FRAMES = 3  # 60 ms
MIC_FLUX_THRESHOLD = 0.12
MIC_ACTIVITY_BLOCK_FRAMES = 5  # 100 ms
MIC_ACTIVITY_SPAN_BLOCKS = 7  # 700 ms
MIC_MIN_ACTIVITY_RATIO = 0.5
MIC_FLUX_WINDOW = np.hanning(FRAME_SAMPLES)[None, :, None]


@dataclass(frozen=True)
class WindowLocation:
    path: Path
    start_sample: int
    end_sample: int


@dataclass
class Bounds:
    low: float | None = None
    high: float | None = None
    low_window: WindowLocation | None = None
    high_window: WindowLocation | None = None
    kept: int = 0
    total: int = 0
    edge_excluded: int = 0
    spectral_vad_excluded: int = 0
    noise_floor_dbfs: float | None = None

    def add(
        self,
        value: float,
        min_dbfs: float,
        location: WindowLocation,
        *,
        edge_only: bool = False,
        spectral_vad_rejected: bool = False,
    ) -> None:
        self.total += 1
        if value < min_dbfs:
            return
        if edge_only:
            self.edge_excluded += 1
            return
        if spectral_vad_rejected:
            self.spectral_vad_excluded += 1
            return
        self.kept += 1
        if self.low is None or value < self.low:
            self.low = value
            self.low_window = location
        if self.high is None or value > self.high:
            self.high = value
            self.high_window = location


def window_dbfs(samples: np.ndarray) -> float:
    """samples: (T, C)。全部通道、全部采样一次均方。"""
    data = np.asarray(samples, dtype=np.float64)
    power = float(np.mean(data * data))
    return float(10.0 * np.log10(power + DBFS_POWER_EPS))


def frame_powers(samples: np.ndarray) -> np.ndarray:
    """将选中通道按 20 ms 切段，返回每段的合并均方功率。"""
    starts = np.arange(0, len(samples), FRAME_SAMPLES)
    lengths = np.diff(starts, append=len(samples))
    sample_power = np.mean(samples * samples, axis=1)
    return np.add.reduceat(sample_power, starts) / lengths


def mic_spectral_features(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 20 ms 帧能量和经时间平滑的正向谱通量。"""
    frame_count = len(samples) // FRAME_SAMPLES
    frames = samples[:frame_count * FRAME_SAMPLES].reshape(frame_count, FRAME_SAMPLES, samples.shape[1])
    spectrum = np.fft.rfft(frames * MIC_FLUX_WINDOW, axis=1)
    power = np.mean(np.abs(spectrum) ** 2, axis=2)
    # 20 ms FFT 的 bin 间距为 50 Hz。合并 200–3950 Hz 内每四个 bin。
    bands = power[:, 4:80].reshape(frame_count, 19, 4).sum(axis=2)
    smoothed = np.lib.stride_tricks.sliding_window_view(bands, MIC_FLUX_SMOOTH_FRAMES, axis=0).mean(axis=-1)
    prev = smoothed[:-MIC_FLUX_LAG_FRAMES]
    curr = smoothed[MIC_FLUX_LAG_FRAMES:]
    positive_change = np.maximum(curr - prev, 0.0).sum(axis=1)
    flux = np.zeros(frame_count, dtype=np.float64)
    flux[MIC_FLUX_SMOOTH_FRAMES - 1 + MIC_FLUX_LAG_FRAMES:] = (
        positive_change / ((prev + curr).sum(axis=1) + DBFS_POWER_EPS)
    )
    return np.mean(frames ** 2, axis=(1, 2)), flux


def mic_has_spectral_activity(features: tuple[np.ndarray, np.ndarray], min_dbfs: float) -> bool:
    powers, flux = features
    energetic = powers + DBFS_POWER_EPS >= 10 ** (min_dbfs / 10)
    active = energetic & (flux >= MIC_FLUX_THRESHOLD)
    complete = len(active) // MIC_ACTIVITY_BLOCK_FRAMES * MIC_ACTIVITY_BLOCK_FRAMES
    # 700 ms 内至少有三个 100 ms 段包含活动，每段至少三个有效帧。
    blocks = active[:complete].reshape(-1, MIC_ACTIVITY_BLOCK_FRAMES).sum(axis=1) >= 3
    spans = np.lib.stride_tricks.sliding_window_view(blocks, MIC_ACTIVITY_SPAN_BLOCKS)
    if not np.any(spans.sum(axis=1) >= 3):
        return False
    # 相邻 100 ms 也可贡献高于底噪门限的帧，保留谱变化较小的元音。
    supported = np.convolve(blocks, (1, 1, 1), mode="same") > 0
    supported_frames = energetic[:complete].reshape(-1, MIC_ACTIVITY_BLOCK_FRAMES)[supported]
    min_frames = math.ceil(len(energetic) * MIC_MIN_ACTIVITY_RATIO)
    return bool(np.count_nonzero(supported_frames) >= min_frames)


def ref_activity_only_at_edge(samples: np.ndarray, min_dbfs: float) -> bool:
    """用 20 ms 均方找活动。活动全在同一端的四分之一窗内（最多 1 秒）时排除。"""
    edge_samples = min(SAMPLE_RATE, len(samples) // 4)
    active = np.flatnonzero(frame_powers(samples) + DBFS_POWER_EPS >= 10 ** (min_dbfs / 10))
    if active.size == 0:
        return False
    first = active[0] * FRAME_SAMPLES
    end = min(len(samples), (active[-1] + 1) * FRAME_SAMPLES)
    return bool(end <= edge_samples or first >= len(samples) - edge_samples)


def collect_wavs(inputs: Sequence[str | Path]) -> list[Path]:
    found: list[Path] = []
    for item in map(Path, inputs):
        if item.is_dir():
            found.extend(item.rglob("*.wav"))
            found.extend(item.rglob("*.WAV"))
        elif item.is_file():
            found.append(item)
        else:
            raise FileNotFoundError(f"路径不存在: {item}")
    unique = list(dict.fromkeys(path.resolve() for path in found))
    if not unique:
        raise ValueError("没有找到 wav")
    return sorted(unique)


def parse_indices(text: str) -> list[int]:
    try:
        values = [int(part) for part in text.split(",") if part != ""]
    except ValueError as exc:
        raise ValueError(f"通道索引必须是逗号分隔的整数: {text}") from exc
    if not values or any(index < 0 for index in values):
        raise ValueError(f"通道索引必须是 0-based 非负整数: {text}")
    if len(set(values)) != len(values):
        raise ValueError(f"通道索引含有重复: {text}")
    return values


def iter_windows(
    path: str | Path,
    indices: Sequence[int],
    window_sec: float,
    hop_sec: float | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """返回重采样后的窗起点与采样。整文件短于一窗时用整段。"""
    if not math.isfinite(window_sec) or window_sec <= 0:
        raise ValueError(f"窗长非法: {window_sec}")
    if hop_sec is None:
        hop_sec = min(DEFAULT_HOP_SEC, window_sec)
    if not math.isfinite(hop_sec) or not 0 < hop_sec <= window_sec:
        raise ValueError(f"步长必须大于 0 且不超过窗长: {hop_sec}")

    window_samples = int(round(window_sec * SAMPLE_RATE))
    if window_samples <= 0:
        raise ValueError(f"窗长非法: {window_sec}")
    hop_samples = int(round(hop_sec * SAMPLE_RATE))
    if hop_samples <= 0:
        raise ValueError(f"步长不足一个采样点: {hop_sec}")

    with sf.SoundFile(path) as audio:
        needed = max(indices) + 1
        if audio.channels < needed:
            raise ValueError(f"{path} 只有 {audio.channels} 通道，索引需要至少 {needed} 通道")
        resampler = None
        if audio.samplerate != SAMPLE_RATE:
            resampler = soxr.ResampleStream(
                audio.samplerate, SAMPLE_RATE, len(indices), dtype="float64", quality="HQ"
            )

        pending = np.empty((0, len(indices)), dtype=np.float64)
        start = 0
        for block in audio.blocks(
            blocksize=audio.samplerate,
            dtype="float64",
            always_2d=True,
        ):
            selected = block[:, indices]
            if resampler is not None:
                selected = resampler.resample_chunk(selected, last=audio.tell() == audio.frames)
            pending = np.concatenate((pending, selected), axis=0)
            offset = 0
            while offset + window_samples <= len(pending):
                yield start, pending[offset:offset + window_samples]
                start += hop_samples
                offset += hop_samples
            pending = pending[offset:]
        if start == 0 and len(pending):
            yield 0, pending


def scan_dbfs_file(
    path: str | Path,
    mic_indices: Sequence[int],
    ref_indices: Sequence[int],
    window_sec: float = DEFAULT_WINDOW_SEC,
    min_dbfs: float = DEFAULT_MIN_DBFS,
    *,
    hop_sec: float | None = None,
    mic_noise_margin_db: float = DEFAULT_MIC_NOISE_MARGIN_DB,
    mic_spectral_vad: bool = True,
) -> tuple[Bounds, Bounds]:
    """扫描单个文件，返回 ``(mic_bounds, ref_bounds)``。

    ``mic_indices`` 和 ``ref_indices`` 都是 0-based、可含多个通道。返回的
    ``Bounds`` 中 ``low`` / ``high`` 为 dBFS，对应窗的位置由
    ``low_window`` / ``high_window`` 给出。
    """
    if not math.isfinite(mic_noise_margin_db) or mic_noise_margin_db < 0:
        raise ValueError(f"MIC 底噪余量必须是有限的非负数: {mic_noise_margin_db}")
    path = Path(path)
    mic_indices = list(mic_indices)
    ref_indices = list(ref_indices)
    if not mic_indices:
        raise ValueError("mic_indices 不能为空")
    if not ref_indices:
        raise ValueError("ref_indices 不能为空")
    if any(index < 0 for index in mic_indices + ref_indices):
        raise ValueError(f"通道索引必须是 0-based 非负整数: {mic_indices + ref_indices}")
    mic_bounds = Bounds()
    ref_bounds = Bounds()
    mic_count = len(mic_indices)
    mic_windows: list[tuple[float, WindowLocation, tuple[np.ndarray, np.ndarray] | None]] = []
    noise_parts: list[np.ndarray] = []
    quiet_end = 0
    min_activity_samples = MIC_ACTIVITY_SPAN_BLOCKS * MIC_ACTIVITY_BLOCK_FRAMES * FRAME_SAMPLES
    for start, block in iter_windows(path, mic_indices + ref_indices, window_sec, hop_sec):
        location = WindowLocation(path, start, start + len(block))
        mic = block[:, :mic_count]
        mic_dbfs = window_dbfs(mic)
        ref = block[:, mic_count:]
        ref_dbfs = window_dbfs(ref)
        edge_only = ref_dbfs >= min_dbfs and ref_activity_only_at_edge(ref, min_dbfs)
        ref_bounds.add(ref_dbfs, min_dbfs, location, edge_only=edge_only)
        features = None
        if ref_dbfs < min_dbfs:
            # 重叠窗的 MIC 采样只计一次，防止同一段底噪重复参与估计。
            powers = frame_powers(mic[max(0, quiet_end - start):])
            noise_parts.append(powers)
            quiet_end = start + len(mic)
            if mic_spectral_vad and mic_dbfs >= min_dbfs and len(mic) >= min_activity_samples:
                features = mic_spectral_features(mic)
        mic_windows.append((mic_dbfs, location, features))

    if not mic_windows:
        raise ValueError(f"空文件: {path}")

    mic_min_dbfs = min_dbfs
    if noise_parts:
        powers = np.concatenate(noise_parts)
        # 排除数字静音，避免静音填充把底噪估计拉到无意义的位置。
        powers = powers[powers + DBFS_POWER_EPS >= 10 ** (min_dbfs / 10)]
        if powers.size >= MIN_NOISE_FRAMES:
            floor_power = float(np.percentile(powers, MIC_NOISE_PERCENTILE))
            mic_bounds.noise_floor_dbfs = 10.0 * math.log10(floor_power + DBFS_POWER_EPS)
            mic_min_dbfs = max(min_dbfs, mic_bounds.noise_floor_dbfs + mic_noise_margin_db)
    for mic_dbfs, location, features in mic_windows:
        rejected = (
            features is not None
            and mic_dbfs >= mic_min_dbfs
            and not mic_has_spectral_activity(features, mic_min_dbfs)
        )
        mic_bounds.add(mic_dbfs, mic_min_dbfs, location, spectral_vad_rejected=rejected)
    return mic_bounds, ref_bounds


def merge_bounds(parts: Sequence[Bounds]) -> Bounds:
    merged = Bounds()
    for part in parts:
        merged.total += part.total
        merged.kept += part.kept
        merged.edge_excluded += part.edge_excluded
        merged.spectral_vad_excluded += part.spectral_vad_excluded
        if part.low is None or part.high is None:
            continue
        if merged.low is None or part.low < merged.low:
            merged.low = part.low
            merged.low_window = part.low_window
        if merged.high is None or part.high > merged.high:
            merged.high = part.high
            merged.high_window = part.high_window
    return merged


def format_bounds(bounds: Bounds) -> str:
    counts = f"{bounds.kept}/{bounds.total} 窗"
    if bounds.edge_excluded:
        counts += f"，排除 {bounds.edge_excluded} 个起停窗"
    if bounds.spectral_vad_excluded:
        counts += f"，谱通量筛除 {bounds.spectral_vad_excluded} 窗"
    if bounds.kept == 0:
        return f"无有效窗（{counts}）"
    return f"[{bounds.low:.1f}, {bounds.high:.1f}] dBFS（{counts}）"


def format_timestamp(sample: int) -> str:
    milliseconds = (sample * 1000 + SAMPLE_RATE // 2) // SAMPLE_RATE
    seconds, milliseconds = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def print_extrema(bounds: Bounds, *, include_path: bool = False) -> None:
    for label, location in (("最低窗", bounds.low_window), ("最高窗", bounds.high_window)):
        if location is None:
            continue
        start = format_timestamp(location.start_sample)
        end = format_timestamp(location.end_sample)
        source = f"  {location.path}" if include_path else ""
        print(f"    {label} {start}–{end}{source}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scan_dbfs",
        description="统一到 16 kHz 扫描多通道录音的均方 dBFS 上下界",
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="wav 文件或包含 wav 的目录")
    parser.add_argument("--mic-indices", required=True, help="0-based 麦克风通道，逗号分隔，如 5,6,7,8")
    parser.add_argument("--ref-indices", required=True, help="0-based 参考通道，逗号分隔，如 9")
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC)
    parser.add_argument(
        "--hop-sec",
        type=float,
        help="扫描步长（秒），默认 1 秒且不超过窗长。设为窗长时不重叠。",
    )
    parser.add_argument(
        "--mic-noise-margin-db",
        type=float,
        default=DEFAULT_MIC_NOISE_MARGIN_DB,
        help="MIC 统计门限高于文件底噪的余量（dB，默认 6）",
    )
    parser.add_argument(
        "--mic-spectral-vad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="筛选 REF 静默的 MIC 窗，谱通量支持的活动须占窗长至少 50%%，至少需 700 ms 可分析音频（默认开启）",
    )
    parser.add_argument(
        "--min-dbfs",
        type=float,
        default=DEFAULT_MIN_DBFS,
        help="静音门限（dBFS），同时用于判断 REF 窗内的短时活动",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    mic_parts: list[Bounds] = []
    ref_parts: list[Bounds] = []
    try:
        mic_indices = parse_indices(args.mic_indices)
        ref_indices = parse_indices(args.ref_indices)
        wavs = collect_wavs(args.inputs)
        for path in wavs:
            mic_bounds, ref_bounds = scan_dbfs_file(
                path,
                mic_indices,
                ref_indices,
                args.window_sec,
                args.min_dbfs,
                hop_sec=args.hop_sec,
                mic_noise_margin_db=args.mic_noise_margin_db,
                mic_spectral_vad=args.mic_spectral_vad,
            )
            mic_parts.append(mic_bounds)
            ref_parts.append(ref_bounds)
            print(path)
            floor = mic_bounds.noise_floor_dbfs
            if floor is None:
                mic_gate = f"底噪参考不足，使用绝对门限 {args.min_dbfs:.1f} dBFS"
            else:
                gate = max(args.min_dbfs, floor + args.mic_noise_margin_db)
                mic_gate = f"底噪估计 {floor:.1f} dBFS，门限 {gate:.1f} dBFS"
            print(f"  mic {format_bounds(mic_bounds)}，{mic_gate}")
            print_extrema(mic_bounds)
            print(f"  ref {format_bounds(ref_bounds)}")
            print_extrema(ref_bounds)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("合计")
    for label, parts in (("mic", mic_parts), ("ref", ref_parts)):
        bounds = merge_bounds(parts)
        print(f"  {label} {format_bounds(bounds)}")
        print_extrema(bounds, include_path=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
