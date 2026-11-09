from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from lzf_utils.audio import scan_dbfs_file
from lzf_utils.audio.scan_dbfs import build_parser, main

FS = 16000


def _tone(fs: int, seconds: float, amplitude: float) -> np.ndarray:
    time = np.arange(round(fs * seconds), dtype=np.float64) / fs
    return amplitude * np.sin(2.0 * np.pi * 440.0 * time)


def _dbfs(samples: np.ndarray) -> float:
    return float(10.0 * np.log10(np.mean(samples * samples)))


def _write_capture(path) -> list[float]:
    levels = [0.001, 0.01, 0.1, 0.2]
    mic = np.concatenate([_tone(FS, 1.0, level) for level in levels])
    ref = np.concatenate(
        [
            np.zeros(FS, dtype=np.float64),
            _tone(FS, 1.0, 0.1),
            _tone(FS, 1.0, 0.1),
            _tone(FS, 1.0, 0.1),
        ]
    )
    sf.write(path, np.stack([mic, ref], axis=1), FS, subtype="FLOAT")
    return levels


def test_scan_dbfs_file_reports_mic_level_range_and_locations(tmp_path) -> None:
    path = tmp_path / "capture.wav"
    levels = _write_capture(path)

    mic_bounds, ref_bounds = scan_dbfs_file(path, [0], [1], window_sec=1.0)

    # 第 0 秒 REF 静默，只用于估计 MIC 底噪；其余三窗都高于门限。
    assert mic_bounds.total == 4
    assert mic_bounds.kept == 3
    assert mic_bounds.low == pytest.approx(_dbfs(_tone(FS, 1.0, levels[1])), abs=0.05)
    assert mic_bounds.high == pytest.approx(_dbfs(_tone(FS, 1.0, levels[3])), abs=0.05)
    assert mic_bounds.low_window is not None
    assert mic_bounds.high_window is not None
    assert mic_bounds.low_window.start_sample == FS
    assert mic_bounds.high_window.start_sample == 3 * FS
    assert ref_bounds.kept == 3
    assert ref_bounds.low == pytest.approx(_dbfs(_tone(FS, 1.0, 0.1)), abs=0.05)


def test_cli_spectral_vad_defaults_on() -> None:
    base = ["--mic-indices", "0", "--ref-indices", "1", "capture.wav"]

    assert build_parser().parse_args(base).mic_spectral_vad is True
    assert build_parser().parse_args([*base, "--no-mic-spectral-vad"]).mic_spectral_vad is False


def test_cli_prints_summary(tmp_path, capsys) -> None:
    path = tmp_path / "capture.wav"
    _write_capture(path)

    assert main(["--mic-indices", "0", "--ref-indices", "1", "--window-sec", "1", str(path)]) == 0

    output = capsys.readouterr().out
    assert "合计" in output
    assert "mic [" in output
    assert "ref [" in output
