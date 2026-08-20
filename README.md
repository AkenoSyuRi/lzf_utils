# lzf-utils

音频 DSP 工具库。按包名导入：

```python
from lzf_utils.audio import AudioWriter, Stft, RoomDataSimulator, PlotUtils
```

作为 git submodule 时，父项目用路径依赖：

```toml
[project]
dependencies = ["lzf-utils"]

[tool.uv.sources]
lzf-utils = { path = "third_party/lzf_utils", editable = true }
```
