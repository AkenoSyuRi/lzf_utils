# AGENTS.md

面向在本仓库工作的 coding agent。人类文档见 `README.md`。

## 项目是什么

`lzf-utils` 是可安装的 Python 工具库，当前主线是音频 DSP（多通道 I/O、逐帧 STFT、房间仿真、可视化）。它会作为其它项目的 git submodule 使用。

- 发行名：`lzf-utils`
- 导入名：`lzf_utils`
- Python：`>=3.12`
- 构建：hatchling，`src/` 布局

父项目应路径依赖安装，不要把仓库根目录塞进 `PYTHONPATH`：

```toml
[project]
dependencies = ["lzf-utils"]

[tool.uv.sources]
lzf-utils = { path = "third_party/lzf_utils", editable = true }
```

```python
from lzf_utils.audio import AudioWriter, Stft, RoomDataSimulator, PlotUtils
```

## 目录

```text
src/lzf_utils/           # 包根；新领域在这里平行建子包
  audio/
    io.py                # 读写、PCM、增益、帧缓冲、RIR 卷积、测试信号
    detect_speech.py     # MATLAB detectSpeech 复刻（能量 + spectral spread VAD）
    stft.py              # 有状态逐帧 STFT / iSTFT
    simulate.py          # pyroomacoustics 房间 / 阵列仿真
    plot.py              # 阵列坐标、语谱图
    __init__.py          # 再导出该领域的公开符号
```

不要把模块再摊回仓库根目录。

## 命令

一律用 `uv`，不要直接调用 `python` / `python3` / `pip`。

```bash
uv sync
uv run python -c "from lzf_utils.audio import AudioWriter"
uv lock
```

改了 `pyproject.toml` 后跑 `uv lock`，不要手改 `uv.lock`。

## 加代码

- 仍属音频：放到 `src/lzf_utils/audio/`，并在 `audio/__init__.py` 再导出公开符号。
- 新领域（例如 `model/`）：在 `src/lzf_utils/<domain>/` 建子包，同样用 `__init__.py` 再导出。
- 包内用相对导入：`from .io import AudioWriter`。
- 对外稳定入口是 `lzf_utils.<domain>`，不要让调用方依赖未导出的内部路径，除非确实在拆内部模块。
- 只实现当前需要的能力。不为一次性逻辑做抽象，不主动加配置项或可选依赖。
- 改动保持最小：不顺手重构、不改无关格式、不删除原有死代码（除非用户明确要求）。

## 依赖

当前直接依赖：`librosa`、`matplotlib`、`numpy`、`pyroomacoustics`、`scipy`、`scipy-stubs`、`soundfile`。

- 新依赖必须先被代码直接 import，再写入 `pyproject.toml`。
- 不要把 `torch` / ONNX 加回默认依赖。若以后需要导出模型，做成 optional extra。
- 不要引入与当前领域无关的通用工具库。

## 风格

- 回复、commit message 默认简体中文；类型名、命令、代码、日志保持原样。
- 跟随现有代码风格（class + `@staticmethod` 也可以，不要为了“现代化”重写）。
- 没有测试套件时，至少用 `uv run python -c` 验证新公开符号可以 import；有行为改动时补最小可运行检查。
