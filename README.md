# lzf-utils

个人音频 DSP 工具库。面向麦克风阵列实验、流式时频处理、房间仿真，以及 Python 原型和固件对拍。

- 发行名：`lzf-utils`
- 导入名：`lzf_utils`
- Python：`>=3.12`
- 包管理：`uv`，hatchling，`src/` 布局

对外一律按包名导入，不要把仓库根目录塞进 `PYTHONPATH`：

```python
from lzf_utils.audio import AudioWriter, Stft, RoomDataSimulator, PlotUtils, detect_speech
```

## 安装

本仓库作为 git submodule 使用时，父项目用路径依赖（建议 editable，改 submodule 立刻生效）：

```toml
[project]
dependencies = ["lzf-utils"]

[tool.uv.sources]
lzf-utils = { path = "third_party/lzf_utils", editable = true }
```

把 submodule 加进父仓库后执行 `uv sync`。

在本仓库内开发：

```bash
uv sync
uv run python -c "from lzf_utils.audio import detect_speech"
```

## 目录

```text
src/lzf_utils/
  audio/
    io.py              # 多通道 wav/pcm 读写、增益、帧缓冲、RIR 卷积、测试信号
    stft.py            # 有状态逐帧 STFT / iSTFT（重叠相加）
    simulate.py        # pyroomacoustics 房间 / 阵列仿真
    plot.py            # 阵列坐标、语谱图
    detect_speech.py   # MATLAB detectSpeech 复刻（能量 + spectral spread VAD）
    rt60/              # RT60 估计、WAV/CLI、诊断绘图
    __init__.py        # 再导出该领域的公开符号
```

新领域（例如模型导出）在 `src/lzf_utils/` 下平行建子包，不要把模块摊回仓库根目录。

## 音频 I/O 与流式处理

`AudioReader` / `AudioWriter` 按帧读写 wav 或 pcm。多通道可以是一个目录里的多个单声道文件，也可以写成一个多通道 wav。

```python
from lzf_utils.audio import AudioReader, AudioWriter, BufferAdapter, StreamingConvolution

reader = AudioReader("mics/", sr=16000)
writer = AudioWriter("out/", sr=16000)

for frames in reader.read_audio_data(frame_len=256):
    # frames: (n_channels, frame_len)
    writer.write_data_list("cap", frames)
```

其它常用符号：

| 符号 | 作用 |
|---|---|
| `AudioUtils.apply_gain` | 按 dB 调整电平 |
| `AudioUtils.pcm2wav` / `wav2pcm` | 裸 PCM 与 WAV 互转 |
| `AudioUtils.merge_channels` | 多路一维信号合成多通道 |
| `BufferAdapter` | 输入帧长和输出帧长不一致时做环形缓冲 |
| `StreamingConvolution` | 按帧和 RIR 做 overlap-save 卷积 |
| `SignalGenerator` | 静音、脉冲、扫频、正弦测试信号 |

## 逐帧 STFT

`Stft` 维护重叠缓冲，适合实时或和 C/固件按 hop 对照。这和 `librosa.stft` 的整段批处理不是同一条路径。

```python
from lzf_utils.audio import Stft
import numpy as np

stft = Stft(fft_size=512, win_size=512, hop_size=256)
frame = np.zeros(256)
spec = stft.transform(frame)
time = stft.inverse(spec)
```

## 房间仿真

`RoomDataSimulator` 用 ShoeBox 房间生成阵列接收信号，可配置 RT60 和 SNR。不传 `rt60` 时为消声；不传 `snr` 时不加噪声。

```python
import numpy as np
from lzf_utils.audio import RoomDataSimulator, SigInfo

mic_pos = np.array([[1.5, 1.6], [2.0, 2.0], [1.2, 1.2]])  # (3, n_mics)
sim = RoomDataSimulator(room_size=[5.0, 4.0, 3.0], mic_pos=mic_pos, fs=16000, rt60=0.3, snr=20)
sig_infos = sim.map2sig_infos(["src.wav"], [np.array([2.0, 3.0, 1.5])])
sim.simulate(*sig_infos)
sim.save("out", "take1", out_db=-20)
```

`get_azimuth_elevation` 以阵列中心为原点，返回方位角 / 俯仰角（默认角度）。`PlotUtils.plot_2d_coord` / `plot_3d_coord` 用来看麦阵几何，`plot_spectrogram` 用来看语谱图。

## 语音检测

`detect_speech` 是 MATLAB Audio Toolbox `detectSpeech` 的 Python 复刻：短时能量 + spectral spread，直方图自动阈值，再按 MergeDistance 合并区间。返回 0-based、右端不包含的 `[start, stop)`，可直接切片 `audio[start:stop]`。

```python
from lzf_utils.audio import detect_speech, read_wav_mono

audio, fs = read_wav_mono("input.wav", mixdown=True)
intervals, thresholds = detect_speech(audio, fs)
for start, stop in intervals:
    print(start, stop, (stop - start) / fs)
```

核心函数默认 30 ms periodic Hann、无重叠，对齐 MATLAB；命令行默认半窗重叠，边界更细。传 `--overlap-ms 0` 可回到 MATLAB 默认。

```bash
uv run python -m lzf_utils.audio.detect_speech input.wav
uv run python -m lzf_utils.audio.detect_speech input.wav --out-wav result.wav --csv result.csv
uv run python -m lzf_utils.audio.detect_speech input.wav --mixdown --plot out.png
```

自动阈值依赖直方图实现，和 MATLAB 可能有小数值差。若要对拍，把 MATLAB 算好的 `thresholds` 传进来即可。

## RT60 估计

`estimate_rt60` 从 RIR 估计 EDT、T20、T30 和最终 RT60。核心流程是 Schroeder 反向积分、Lundeby 噪声截断，以及在 -5..-25 dB / -5..-35 dB 区间内做最小二乘拟合。

```python
from lzf_utils.audio import estimate_rt60, estimate_rt60_file

result = estimate_rt60(rir, fs=16000, method="auto", bands="octave")
channel = result.channels[0]
print(channel.rt60, channel.method_used, channel.valid, channel.reason)
print(channel.edt, channel.t20, channel.t30)
print(result.stats.median, result.stats.mean, result.stats.std)

result = estimate_rt60_file("room_rir.wav")
```

诊断图也可以直接从 API 生成：

```python
from lzf_utils.audio import plot_rt60

plot_rt60(result, channel=0, output_path="rt60_diagnostics.png")
```

- `rir`：`(N,)` 或 `(N, C)`，默认时间轴为 0，可用 `time_axis` 指定。多通道逐通道独立处理，不会先平均波形。
- `method`：`auto` / `EDT` / `T20` / `T30`。`auto` 优先使用可信的 T30，其次 T20；不会用 EDT 顶替。
- `bands`：`None` 关闭分频带；`"octave"` 使用 125、250、500、1000、2000、4000 Hz；也可传自定义中心频率序列。超过 Nyquist 的频带会被跳过。
- `detect_onset`：默认开启直达声起点检测；关闭后从第 0 个样本开始积分。
- `min_r2`：拟合优度阈值，默认 0.95。
- `min_noise_margin_db`：拟合终点与噪声底的最小余量，默认 10 dB。
- `max_method_disagreement`：`auto` 回退到 T20 时允许的 T20/T30 相对差异，默认 0.25；超过时判为无效。

每个通道的结果包含 `edt`、`t20`、`t30`、`rt60`、`method_used`、`r2`、`valid`、`reason`、`decay_range_db`、`onset_sample/time_s`，以及：

- `methods`：每种方法的拟合区间、斜率、R²、噪声污染和噪声余量；
- `noise`：噪声功率、峰值信噪比、EDC 噪声底、截断位置、迭代次数；
- `edc_time_s` / `edc_db`：用于绘图和调试的 EDC 数据。

无效结果使用 `rt60=NaN`、`valid=False`，并在 `reason` 中给出原因；导出 JSON 时非有限值写成 `null`。各方法自己的诊断数值仍会保留，便于分析为什么不可信。

### 噪声处理

1. 用 RIR 尾部 10% 估计初始噪声功率；
2. 对平方 RIR 分块平均，在“噪声 + 10 dB”到“噪声 + 30 dB”的衰减段拟合直线；
3. 求直线与噪声底的交点，并按衰减速度自适应选择块长，迭代更新噪声和交点；
4. 在交点处截断 Schroeder 积分。

这里只做截断，不做尾部能量补偿：单指数尾部的补偿项对 EDC 是常数，不改变斜率；拟合区间又要求在截断点上方保留 10 dB 余量。这样比无条件补偿更不容易在低信噪比时制造虚假的衰减。若噪声底无法可靠估计，或截断点太早，结果直接判无效，不会退化成完整含噪 RIR 的强制拟合。

宽带处理只做常量峰值归一化，不做 AGC、动态范围压缩或随时间的增益处理；分频带只使用零相位 SOS 带通滤波。

### 命令行

```bash
uv run rt60 room_rir.wav
uv run rt60 room_rir.wav --method T20 --bands octave --plot --output result.json
uv run python -m lzf_utils.audio.rt60 room_rir.wav
```

`--plot` 默认把诊断图写到输入文件同目录的 `<stem>_rt60.png`。`--bands octave` 会额外输出各频带结果；`--channel` 控制绘图使用哪个通道。

需要把 EDC 也写进 JSON 时，用 `result.to_json("result.json", include_edc=True)`。

### 验证

在本机 100 个仿真 RIR（5 声源 × 16 麦克风，共 8000 通道）上，以仿真标注的 Sabine RT60 为参考：

| 项目 | 有效数 | 中位误差 | 平均绝对误差 | 90% 绝对误差 |
|---|---:|---:|---:|---:|
| T30 | 7714/8000 | +0.38% | 5.60% | 13.05% |
| auto 最终值 | 7953/8000 | +0.61% | 5.97% | 14.23% |
| T20 | 7682/8000 | -0.31% | 8.22% | 17.99% |

EDT 的偏差更大（中位约 -13%），且大量近场通道的 EDT 拟合 R² 不足；这是因为强直达声主导了 0 到 -10 dB 的早期 EDC。近场/远场声源的 T30 平均绝对误差分别在 4.6% 到 6.6% 之间。与 `pyroomacoustics.experimental.measure_rt60` 在 1583 个有效通道上交叉验证，相关系数约 0.98；后者用两个阈值交点外推且不处理噪声/起点，中位误差约 -8.3%，说明分段最小二乘拟合更适合这里的仿真 RIR。

### 可信度与限制

- T30 至少要能从 -5 dB 拟合到 -35 dB，并且拟合终点比噪声底高 `min_noise_margin_db`。不满足时 `auto` 会尝试 T20；T20 也不满足则判无效。回退到 T20 时还会检查 T20 与 T30 的差异，超过 `max_method_disagreement` 的结果会被判无效。
- 结果同时检查衰减范围、斜率方向、R² 和噪声余量，不会只看 R²。
- 近场/强直达声可能让 EDT 很短，也会让 T30 偏离房间的 Sabine RT60；这是 RIR 位置相关的正常现象，建议用多个传声器位置统计。
- 频带能量过低、频带长度不足或噪声底不可靠时，对应频带会被标记无效。
- 非指数衰减（耦合房间、频率相关吸收、非平稳噪声）不满足单斜率模型，T20/T30 差异大或 R² 低时应谨慎。
- 输入应是去除明显 DC 偏置的 RIR；本工具不做额外的宽带高通或去趋势处理。语音、音乐等非 RIR 信号即使返回数值也没有物理意义。
- 明显削波的 RIR 会破坏指数衰减假设；工具会通过衰减范围、R² 和噪声检查尽量识别，但不专门做削波检测。
- 本工具是 RIR 后处理工具，不替代 ISO 3382 的测量规范；多通道统计只在逐通道估计完成后进行。

## 依赖

`librosa`、`matplotlib`、`numpy`、`pyroomacoustics`、`scipy`、`soundfile`。不要把 `torch` / ONNX 加进默认依赖。
