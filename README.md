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

## 依赖

`librosa`、`matplotlib`、`numpy`、`pyroomacoustics`、`scipy`、`soundfile`。不要把 `torch` / ONNX 加进默认依赖。
