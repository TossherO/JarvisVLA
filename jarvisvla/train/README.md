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

## 三阶段统一框架（新增）

当前训练入口支持通过统一的 stage 配置文件启动三阶段训练。

- 新增配置目录：`configs/stages/`
  - `stage1_qwen2_vl_7b.json`
  - `stage2_qwen2_vl_7b.json`
  - `stage3_qwen2_vl_7b.json`
- 统一启动脚本：`scripts/train/run_stage.sh`
- 多机多卡启动脚本：`scripts/train/run_stage_multinode.sh`

`train.py` 新增支持：

- `--stage_name`：阶段名（`stage1`/`stage2`/`stage3`）
- `--stage_config_path`：阶段配置 JSON 路径
- `--strict_stage_config`：是否对配置中的未知字段报错

配置文件支持四类参数命名空间：

- `script_arguments`：对应 `ScriptArguments`（例如 `dataset_name`）
- `training_arguments`：对应 `SFTConfig` / `TrainingArguments`
- `model_arguments`：对应 `ModelConfig`
- `more_arguments`：对应 `MoreConfig`

此外，`MoreConfig` 新增：

- `train_split`、`eval_split`：可配置训练/验证 split 名称
- `image_folder`：可显式覆盖 collator 的图像根目录

示例：

```bash
bash scripts/train/run_stage.sh configs/stages/stage1_qwen2_vl_7b.json
bash scripts/train/run_stage.sh configs/stages/stage2_qwen2_vl_7b.json
bash scripts/train/run_stage.sh configs/stages/stage3_qwen2_vl_7b.json
bash scripts/train/run_stage.sh configs/stages/stage3_qwen2_vl_7b_test.json

# 多机多卡（使用 hostfile）
bash scripts/train/run_stage_multinode.sh configs/stages/stage3_qwen2_vl_7b.json

# 可选：自定义 hostfile
bash scripts/train/run_stage_multinode.sh configs/stages/stage3_qwen2_vl_7b.json scripts/train/hostfile

# 可选：显式指定多机多卡映射
DEEPSPEED_INCLUDE="hgx1:0,1,2,3@hgx2:0,1,2,3" \
  bash scripts/train/run_stage_multinode.sh configs/stages/stage3_qwen2_vl_7b.json
```

`train.py` 会在参数解析前读取 `--stage_config_path` 指向的 JSON，并将其中的默认参数注入 CLI（显式 CLI 参数优先级更高），因此 bash 脚本无需再预解析配置内容。

`run_stage.sh` 当前行为：

- 只接受一个位置参数：`<config_path>`。
- 默认 `WANDB_MODE=offline`，如需在线同步可手动设置 `WANDB_MODE=online`。
- 未设置 `CUDA_VISIBLE_DEVICES` 时，会自动选择当前机器上的全部可见 GPU。

`run_stage_multinode.sh` 当前行为：

- 接受参数：`<config_path> [hostfile_path]`。
- 默认 hostfile：`scripts/train/hostfile`。
- 默认 `WANDB_MODE=offline`。
- 可通过 `DEEPSPEED_INCLUDE` 显式传入节点与 GPU 映射。
