# jarvisvla/train 目录说明

本目录负责训练阶段逻辑，目标是将多模态样本组织为模型可学习格式，并完成 SFT 微调。

## 文件功能

- `__init__.py`
  - 包初始化文件。

- `train.py`
  - 训练主入口。
  - 解析训练参数（TRL/Transformers）。
  - 加载 Qwen2-VL 模型与 processor。
  - 注入 Minecraft 特殊 token。
  - 按配置冻结视觉或语言子模块参数。
  - 构建数据 collator，加载数据集并启动 `Trainer`。

- `data_collator.py`
  - 数据整理核心模块。
  - 将对话内容中的文本、图像、点位、框标注转换成模型输入。
  - 负责图像增强与 resize。
  - 构造 labels 并屏蔽不参与训练的 token（如 user 段和 padding）。

- `utils_train.py`
  - 训练辅助函数与配置。
  - 定义 `MoreConfig`（如冻结策略、像素范围、采样比例）。
  - 提供随机种子设置与可训练参数统计导出。

## 训练数据假设

训练数据按对话样式组织，常见字段包括：

- 对话 `conversations`
- 图像路径或图像字节
- 样本 id

`data_collator.py` 会将这些信息统一处理后输出给 `Trainer`。
