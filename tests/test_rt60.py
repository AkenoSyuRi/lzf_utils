from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy import signal

from lzf_utils.audio import estimate_rt60, estimate_rt60_file
from lzf_utils.audio.rt60.cli import main


def make_rir(
    rt60: float,
    *,
    fs: int = 16000,
    duration_factor: float = 1.5,
    seed: int = 0,
    noise_snr_db: float | None = None,
) -> np.ndarray:
    """生成幅度包络严格为指数衰减的随机 RIR。"""

    rng = np.random.default_rng(seed)
    n_samples = round(fs * rt60 * duration_factor)
    time = np.arange(n_samples, dtype=np.float64) / fs
    rir = rng.standard_normal(n_samples) * 10.0 ** (-3.0 * time / rt60)
    rir /= np.max(np.abs(rir))
    if noise_snr_db is not None:
        noise_rms = 10.0 ** (-noise_snr_db / 20.0)
        rir = rir + noise_rms * rng.standard_normal(n_samples)
    return rir


def naive_rt60_from_full_edc(rir: np.ndarray, fs: int) -> float:
    """不做噪声截断、直接在完整 EDC 上拟合 -5..-35 dB 的简单实现。"""

    energy = np.cumsum(np.square(rir[::-1], dtype=np.float64))[::-1]
    edc_db = 10.0 * np.log10(np.maximum(energy / energy[0], 1e-30))
    start = int(np.flatnonzero(edc_db <= -5.0)[0])
    candidates = np.flatnonzero(edc_db <= -35.0)
    end = int(candidates[0]) if candidates.size else edc_db.size - 1
    time = np.arange(edc_db.size, dtype=np.float64) / fs
    slope = np.polyfit(time[start : end + 1], edc_db[start : end + 1], 1)[0]
    return -60.0 / slope


@pytest.mark.parametrize("rt60", [0.3, 0.6, 1.0])
def test_ideal_exponential_rir(rt60: float) -> None:
    rir = make_rir(rt60, seed=0)
    result = estimate_rt60(rir, fs=16000)

    channel = result.channels[0]
    assert channel.valid
    assert channel.method_used == "T30"
    assert abs(channel.t20 / rt60 - 1.0) < 0.10
    assert abs(channel.t30 / rt60 - 1.0) < 0.10
    assert channel.broadband.decay_range_db > 40.0
    assert channel.broadband.noise.noise_floor_db < -40.0
    assert np.all(np.isfinite(channel.broadband.edc_db))

    edc = channel.broadband.edc_db
    assert edc[0] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(edc) <= 1e-9)
    assert channel.broadband.edc_time_s[0] == 0.0
    assert channel.broadband.edc_time_s.size == edc.size


def test_noise_truncation_beats_naive_full_integration() -> None:
    rt60 = 0.6
    rir = make_rir(rt60, seed=42, noise_snr_db=50.0)
    result = estimate_rt60(rir, fs=16000)
    channel = result.channels[0]

    assert channel.valid
    assert channel.method_used == "T20"
    assert abs(channel.rt60 / rt60 - 1.0) < 0.10
    assert not channel.broadband.methods["T30"].valid
    assert channel.broadband.methods["T30"].reason == "noise_margin"

    naive = naive_rt60_from_full_edc(rir, 16000)
    assert abs(naive / rt60 - 1.0) > 0.5


def test_too_much_noise_returns_invalid_not_number() -> None:
    rt60 = 0.6
    rir = make_rir(rt60, seed=42, noise_snr_db=30.0)
    result = estimate_rt60(rir, fs=16000)
    channel = result.channels[0]

    assert not channel.valid
    assert math.isnan(channel.rt60)
    assert channel.method_used is None
    assert channel.reason
    assert not channel.broadband.methods["T20"].valid
    assert not channel.broadband.methods["T30"].valid


def test_auto_rejects_t20_t30_disagreement() -> None:
    fs = 16000
    rng = np.random.default_rng(0)
    n_samples = round(1.2 * fs)
    time = np.arange(n_samples, dtype=np.float64) / fs
    rir = (
        rng.standard_normal(n_samples) * 10.0 ** (-3.0 * time / 0.15)
        + 0.05 * rng.standard_normal(n_samples) * 10.0 ** (-3.0 * time / 0.6)
    )
    rir /= np.max(np.abs(rir))

    result = estimate_rt60(rir, fs=fs)
    channel = result.channels[0]

    assert not channel.valid
    assert channel.reason is not None and channel.reason.startswith("method_disagreement")
    assert channel.broadband.methods["T20"].valid
    assert not channel.broadband.methods["T30"].valid


def test_pure_noise_is_invalid() -> None:
    rng = np.random.default_rng(7)
    result = estimate_rt60(rng.standard_normal(16000), fs=16000)
    channel = result.channels[0]

    assert not channel.valid
    assert math.isnan(channel.rt60)
    assert "noise_estimation_failed" in (channel.reason or "")


def test_multichannel_is_independent_and_stats_exclude_invalid() -> None:
    rir_a = make_rir(0.4, seed=1)
    rir_b = -make_rir(0.4, seed=1)
    rir = np.stack([rir_a, rir_b, np.zeros_like(rir_a)], axis=1)

    result = estimate_rt60(rir, fs=16000)

    assert result.n_channels == 3
    assert result.channels[0].valid
    assert result.channels[1].valid
    assert not result.channels[2].valid
    assert result.channels[2].reason == "all_zero"
    assert result.stats.n_valid == 2
    assert result.stats.n_total == 3
    assert abs(result.channels[0].rt60 - result.channels[1].rt60) < 1e-12
    assert result.stats.median == pytest.approx(result.channels[0].rt60)
    assert result.stats.std == pytest.approx(0.0)


def test_time_axis_one_matches_time_axis_zero() -> None:
    rir = make_rir(0.5, seed=3)
    result_0 = estimate_rt60(rir, fs=16000, time_axis=0)
    result_1 = estimate_rt60(rir[None, :], fs=16000, time_axis=1)

    assert result_1.channels[0].rt60 == pytest.approx(result_0.channels[0].rt60)
    assert result_1.channels[0].broadband.onset_sample == result_0.channels[0].broadband.onset_sample


def test_leading_silence_and_onset_switch() -> None:
    fs = 16000
    rir = make_rir(0.5, fs=fs, seed=4)
    padded = np.concatenate([np.zeros(1600), rir])

    result = estimate_rt60(padded, fs=fs)
    channel = result.channels[0]
    assert channel.valid
    assert abs(channel.rt60 / 0.5 - 1.0) < 0.10
    assert channel.broadband.onset_sample >= 1580

    result_no_detect = estimate_rt60(padded, fs=fs, detect_onset=False)
    channel_no_detect = result_no_detect.channels[0]
    assert channel_no_detect.broadband.onset_sample == 0
    assert channel_no_detect.valid
    assert abs(channel_no_detect.rt60 / 0.5 - 1.0) < 0.10


@pytest.mark.parametrize("fs", [8000, 16000, 44100, 48000])
def test_supported_sampling_rates(fs: int) -> None:
    result = estimate_rt60(make_rir(0.5, fs=fs, seed=5), fs=fs, bands="octave")
    channel = result.channels[0]

    assert channel.valid
    assert abs(channel.rt60 / 0.5 - 1.0) < 0.10
    if fs == 8000:
        assert 4000.0 not in result.bands_hz
        assert 2000.0 in result.bands_hz


def test_octave_bands_and_low_energy_band() -> None:
    fs = 16000
    rng = np.random.default_rng(2)
    n_samples = round(0.75 * fs)
    time = np.arange(n_samples, dtype=np.float64) / fs
    noise = rng.standard_normal(n_samples)
    noise -= noise.mean()
    lowpass = signal.butter(6, 500.0, btype="lowpass", fs=fs, output="sos")
    low = signal.sosfiltfilt(lowpass, noise)
    fade = np.minimum(1.0, time / 0.01)
    rir = low * np.exp(-time / 0.0724) * fade
    rir /= np.max(np.abs(rir))

    result = estimate_rt60(rir, fs=fs, bands="octave")
    assert result.bands_hz == (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0)
    for center in (125.0, 250.0, 500.0):
        assert result.channels[0].bands[center].valid
    assert not result.channels[0].bands[4000.0].valid
    assert result.channels[0].bands[4000.0].reason == "band_energy_too_low"
    assert result.bands[4000.0].stats.n_valid == 0


@pytest.mark.parametrize("method", ["EDT", "T20", "T30"])
def test_explicit_method_selection(method: str) -> None:
    rt60 = 0.6
    result = estimate_rt60(make_rir(rt60, seed=6), fs=16000, method=method)
    channel = result.channels[0]

    assert channel.valid
    assert channel.method_used == method
    assert abs(channel.rt60 / rt60 - 1.0) < 0.10


@pytest.mark.parametrize(
    "bad_input",
    [
        [],
        np.array([np.nan, 0.0, 1.0]),
        np.array([0.0, np.inf, 1.0]),
        np.array([1.0 + 1.0j, 0.0]),
        np.zeros((2, 3, 4)),
    ],
)
def test_invalid_inputs_raise(bad_input: object) -> None:
    with pytest.raises(ValueError):
        estimate_rt60(bad_input, fs=16000)


@pytest.mark.parametrize("fs", [0.0, -1.0, math.nan])
def test_invalid_sampling_rate_raises(fs: float) -> None:
    with pytest.raises(ValueError):
        estimate_rt60(make_rir(0.5), fs=fs)


def test_short_and_silent_inputs_are_invalid_not_exceptions() -> None:
    short = estimate_rt60(np.ones(100), fs=16000).channels[0]
    assert not short.valid
    assert short.reason == "rir_too_short"
    assert math.isnan(short.rt60)

    silent = estimate_rt60(np.zeros(16000), fs=16000).channels[0]
    assert not silent.valid
    assert silent.reason == "all_zero"
    assert math.isnan(silent.rt60)

    short_with_bands = estimate_rt60(np.ones(20), fs=16000, bands="octave")
    assert not short_with_bands.channels[0].valid
    for center in short_with_bands.bands_hz:
        band = short_with_bands.channels[0].bands[center]
        assert not band.valid
        assert band.reason == "rir_too_short"

    silent_with_bands = estimate_rt60(np.zeros(16000), fs=16000, bands="octave")
    for center in silent_with_bands.bands_hz:
        band = silent_with_bands.channels[0].bands[center]
        assert not band.valid
        assert band.reason == "all_zero"


def test_json_serialization_is_strict() -> None:
    result = estimate_rt60(make_rir(0.5, seed=8), fs=16000, bands="octave")
    text = result.to_json()
    data = json.loads(text)
    assert data["n_channels"] == 1
    assert data["channels"][0]["broadband"]["valid"] is True
    assert data["bands_hz"] == [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]
    json.dumps(data, allow_nan=False)

    invalid = estimate_rt60(np.zeros(16000), fs=16000)
    invalid_text = invalid.to_json()
    invalid_data = json.loads(invalid_text)
    assert invalid_data["channels"][0]["broadband"]["rt60"] is None
    json.dumps(invalid_data, allow_nan=False)

    with_edc = json.loads(result.to_json(include_edc=True))
    assert len(with_edc["channels"][0]["broadband"]["edc_db"]) > 0
    assert with_edc["channels"][0]["broadband"]["edc_time_s"][0] == 0.0


def test_wav_api_cli_and_plot(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fs = 16000
    rir = make_rir(0.4, fs=fs, seed=9)
    wav_path = tmp_path / "rir.wav"
    sf.write(wav_path, rir.astype(np.float32), fs)

    from_array = estimate_rt60(rir, fs=fs)
    from_file = estimate_rt60_file(wav_path)
    assert from_file.channels[0].valid
    assert from_file.channels[0].rt60 == pytest.approx(from_array.channels[0].rt60, abs=5e-3)

    json_path = tmp_path / "result.json"
    assert main([str(wav_path), "--bands", "octave", "--output", str(json_path)]) == 0
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["n_channels"] == 1
    assert data["channels"][0]["broadband"]["rt60"] == pytest.approx(
        from_file.channels[0].rt60,
        abs=1e-6,
    )

    assert main([str(wav_path), "--plot"]) == 0
    assert (tmp_path / "rir_rt60.png").exists()
    captured = capsys.readouterr()
    assert "RT60" in captured.out


def test_python_dash_m_entry_point(tmp_path: Path) -> None:
    fs = 16000
    wav_path = tmp_path / "rir.wav"
    sf.write(wav_path, make_rir(0.4, fs=fs, seed=10).astype(np.float32), fs)
    json_path = tmp_path / "result.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "lzf_utils.audio.rt60",
            str(wav_path),
            "--output",
            str(json_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["channels"][0]["broadband"]["valid"] is True
