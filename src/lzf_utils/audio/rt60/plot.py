"""RT60 诊断绘图。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .estimate import MethodName, RT60Estimate, RT60Result

_METHOD_COLORS = {"EDT": "C2", "T20": "C1", "T30": "C0"}


def _plot_waveform(ax: Axes, result: RT60Result, channel: int, estimate: RT60Estimate) -> None:
    signal_data = result.rir[:, channel]
    time = np.arange(signal_data.size, dtype=np.float64) / result.fs
    ax.plot(time, signal_data, color="0.2", linewidth=0.7)
    ax.axvline(estimate.onset_time_s, color="C3", linestyle="--", linewidth=1.0, label="onset")
    end_time = estimate.onset_time_s + estimate.noise.truncation_time_s
    if estimate.noise.truncated:
        ax.axvline(end_time, color="0.4", linestyle=":", linewidth=1.0, label="truncation")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("RIR waveform")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)


def _plot_edc(ax: Axes, estimate: RT60Estimate) -> None:
    if estimate.edc_db.size == 0:
        ax.text(0.5, 0.5, "EDC unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    ax.plot(estimate.edc_time_s, estimate.edc_db, color="0.15", linewidth=1.0, label="EDC")
    for method in ("EDT", "T20", "T30"):
        item = estimate.methods[method]
        if not np.isfinite(item.fit_start_s) or not np.isfinite(item.fit_end_s):
            continue
        color = _METHOD_COLORS[method]
        ax.axvspan(item.fit_start_s, item.fit_end_s, color=color, alpha=0.08, linewidth=0)
        ax.plot(
            [item.fit_start_s, item.fit_end_s],
            [item.fit_start_db, item.fit_end_db],
            color=color,
            linestyle="--",
            linewidth=1.5,
            label=f"{method} fit",
        )

    if np.isfinite(estimate.noise.noise_floor_db):
        ax.axhline(
            estimate.noise.noise_floor_db,
            color="0.5",
            linestyle=":",
            linewidth=1.0,
            label="noise floor",
        )
    if estimate.noise.truncated:
        ax.axvline(
            estimate.noise.truncation_time_s,
            color="0.3",
            linestyle=":",
            linewidth=1.0,
            label="truncation",
        )

    ax.set_xlabel("Time from onset (s)")
    ax.set_ylabel("EDC (dB)")
    method = estimate.method_used or "invalid"
    if estimate.valid:
        title = f"EDC  RT60={estimate.rt60:.3f} s ({method})"
    else:
        title = f"EDC  invalid ({estimate.reason})"
    ax.set_title(title, fontsize=10)
    ax.set_xlim(left=0.0)
    if np.isfinite(estimate.noise.noise_floor_db):
        bottom = max(-100.0, min(-45.0, estimate.noise.noise_floor_db - 5.0))
    else:
        bottom = -80.0
    ax.set_ylim(bottom=bottom, top=5.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)


def _plot_band_rt60(ax: Axes, result: RT60Result, channel: int) -> None:
    if not result.bands_hz:
        ax.set_axis_off()
        return

    estimates = [result.channels[channel].bands[center] for center in result.bands_hz]
    labels = [f"{center:g}" for center in result.bands_hz]
    values = [item.rt60 if item.valid else 0.0 for item in estimates]
    colors = ["C0" if item.valid else "0.75" for item in estimates]
    ax.bar(labels, values, color=colors)
    for index, item in enumerate(estimates):
        if item.valid:
            ax.text(index, item.rt60, f"{item.rt60:.3f}", ha="center", va="bottom", fontsize=8)
        else:
            ax.text(index, 0.02, "invalid", ha="center", va="bottom", rotation=90, fontsize=7, color="C3")
    ax.set_xlabel("Octave band center (Hz)")
    ax.set_ylabel("RT60 (s)")
    ax.set_title("Band RT60")
    ax.grid(True, axis="y", alpha=0.3)


def _plot_methods(ax: Axes, estimate: RT60Estimate) -> None:
    methods: tuple[MethodName, ...] = ("EDT", "T20", "T30")
    values = [estimate.methods[name].rt60 for name in methods]
    colors = [
        _METHOD_COLORS[name] if estimate.methods[name].valid else "0.75"
        for name in methods
    ]
    positions = np.arange(len(methods), dtype=np.float64)
    heights = [value if np.isfinite(value) else 0.0 for value in values]
    ax.bar(positions, heights, color=colors, width=0.6)
    for index, (name, value) in enumerate(zip(methods, values)):
        if np.isfinite(value):
            ax.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
        else:
            ax.text(index, 0.01, "invalid", ha="center", va="bottom", rotation=90, fontsize=7, color="C3")
    ax.set_xticks(positions, methods)
    ax.set_ylabel("RT60 (s)")
    ax.set_title("Method comparison (broadband)")
    ax.grid(True, axis="y", alpha=0.3)
    if estimate.valid:
        ax.axhline(estimate.rt60, color="0.2", linestyle="--", linewidth=1.0, label="selected")
        ax.legend(loc="upper left", fontsize=8)


def plot_rt60(
    result: RT60Result,
    channel: int = 0,
    output_path: str | Path | None = None,
    *,
    show: bool = False,
) -> Figure:
    """绘制 RIR、EDC 拟合、噪声截断、方法对比和分频带 RT60。

    Parameters
    ----------
    result:
        :func:`estimate_rt60` 的返回值。
    channel:
        要绘制的通道号。
    output_path:
        如果提供，保存为图片。
    show:
        是否调用 ``matplotlib.pyplot.show()``；默认不阻塞。

    Returns
    -------
    matplotlib.figure.Figure
    """

    if not 0 <= channel < result.n_channels:
        raise IndexError(f"channel 超出范围：{channel}，有效范围 0..{result.n_channels - 1}")

    estimate = result.channels[channel].broadband
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))

    _plot_waveform(axes[0, 0], result, channel, estimate)
    _plot_edc(axes[0, 1], estimate)
    _plot_band_rt60(axes[1, 0], result, channel)
    _plot_methods(axes[1, 1], estimate)

    figure.suptitle(f"RT60 diagnostics - channel {channel}")
    figure.tight_layout()

    if output_path is not None:
        figure.savefig(Path(output_path), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return figure
