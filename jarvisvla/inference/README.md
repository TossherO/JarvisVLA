# jarvisvla/inference 目录说明

本目录负责推理侧的核心桥接：把视觉与文本输入组织成模型请求，并将模型输出 token 转回 Minecraft 动作。

## 文件功能

- `__init__.py`
  - 包初始化文件。

- `load_model.py`
  - 根据 checkpoint 名称判断并返回当前使用的模型骨干类型。
  - 当前主要适配 `qwen2_vl`。

- `processor_wrapper.py`
  - 推理输入封装器。
  - 提供图像 resize、base64 编码、消息结构构建（OpenAI/vLLM 风格）。
  - 统一图像与文本 prompt 的组织方式。

- `action_mapping.py`
  - 动作 token 编码/解码核心。
  - 定义控制 token 与 Minecraft 动作空间映射关系。
  - 提供 `ActionTokenizer` 体系，将模型输出转换为环境可执行动作。

- `construct.py`
  - 用于导出完整模型资源（尤其是 processor 和特殊 token）。
  - 面向部署阶段，确保训练端与推理端配置一致。

## 目录作用总结

`inference/` 是“模型输出到环境动作”的关键中间层：

1. 输入侧：图像与文字 -> 标准消息格式。
2. 输出侧：token 序列 -> Minecraft 动作序列。
