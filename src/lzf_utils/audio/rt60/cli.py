"""RT60 命令行工具。"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path

from .estimate import RT60Result
from .io import estimate_rt60_file


def _format_value(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def _print_summary(result: RT60Result) -> None:
    print(f"fs = {result.fs:g} Hz, channels = {result.n_channels}, method = {result.method}")
    for channel in result.channels:
        item = channel.broadband
        if item.valid:
            print(
                f"ch {channel.channel}: RT60 = {_format_value(item.rt60)} s "
                f"({item.method_used}, R2 = {_format_value(item.r2, 4)})"
            )
        else:
            print(f"ch {channel.channel}: invalid ({item.reason})")
        print(
            "    "
            f"EDT={_format_value(item.edt)}  T20={_format_value(item.t20)}  "
            f"T30={_format_value(item.t30)}  decay={_format_value(item.decay_range_db, 1)} dB  "
            f"noise_floor={_format_value(item.noise.noise_floor_db, 1)} dB  "
            f"trunc={_format_value(item.noise.truncation_time_s)} s"
        )

    stats = result.stats
    print(
        "channel RT60: "
        f"median={_format_value(stats.median)}  mean={_format_value(stats.mean)}  "
        f"std={_format_value(stats.std)}  valid={stats.n_valid}/{stats.n_total}"
    )
    for name in ("EDT", "T20", "T30"):
        method_stats = result.method_stats[name]  # type: ignore[index]
        print(
            f"{name}: median={_format_value(method_stats.median)}  "
            f"valid={method_stats.n_valid}/{method_stats.n_total}"
        )

    if result.bands_hz:
        parts = []
        for center in result.bands_hz:
            band_stats = result.bands[center].stats
            parts.append(f"{center:g}Hz={_format_value(band_stats.median)}")
        print("band RT60 median: " + ", ".join(parts))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rt60",
        description="从 RIR 估计 RT60（Schroeder EDC + Lundeby 噪声截断）。",
    )
    parser.add_argument("input", type=Path, help="RIR 文件（WAV/FLAC 等，soundfile 支持即可）")
    parser.add_argument(
        "--method",
        choices=("auto", "EDT", "T20", "T30"),
        default="auto",
        help="估计方法，默认 auto（T30 -> T20）",
    )
    parser.add_argument(
        "--bands",
        choices=("none", "octave"),
        default="none",
        help="分频带模式，默认 none",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="保存诊断图到输入文件同目录的 <stem>_rt60.png",
    )
    parser.add_argument("--output", type=Path, default=None, help="JSON 输出路径")
    parser.add_argument("--channel", type=int, default=0, help="绘图使用的通道号，默认 0")
    parser.add_argument("--min-r2", type=float, default=0.95, help="最小拟合 R²，默认 0.95")
    parser.add_argument(
        "--min-noise-margin",
        type=float,
        default=10.0,
        help="拟合终点与噪声底的最小余量（dB），默认 10",
    )
    parser.add_argument(
        "--max-method-disagreement",
        type=float,
        default=0.25,
        help="auto 回退 T20 时允许的 T20/T30 相对差异，默认 0.25",
    )
    parser.add_argument(
        "--no-onset-detect",
        action="store_true",
        help="关闭直达声起点检测",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        result = estimate_rt60_file(
            args.input,
            method=args.method,
            bands=args.bands,
            detect_onset=not args.no_onset_detect,
            min_r2=args.min_r2,
            min_noise_margin_db=args.min_noise_margin,
            max_method_disagreement=args.max_method_disagreement,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    _print_summary(result)

    if args.plot and not 0 <= args.channel < result.n_channels:
        print(
            f"error: --channel {args.channel} 超出范围 0..{result.n_channels - 1}",
            file=sys.stderr,
        )
        return 2

    try:
        if args.output is not None:
            result.to_json(args.output)
            print(f"JSON -> {args.output}")

        if args.plot:
            from matplotlib import pyplot as plt

            from .plot import plot_rt60

            output_path = args.input.with_name(f"{args.input.stem}_rt60.png")
            figure = plot_rt60(result, channel=args.channel, output_path=output_path)
            plt.close(figure)
            print(f"plot -> {output_path}")
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
