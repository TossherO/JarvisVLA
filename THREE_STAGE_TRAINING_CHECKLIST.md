# JarvisVLA 三阶段训练补齐工程清单

本文档基于当前仓库代码和论文中的三阶段描述，整理如果要把完整训练流程补齐，工程上还需要增加什么。

## 一、当前仓库的实际情况

当前代码更接近一个统一的 post-training / SFT 流程：

- 训练入口只有一条主线，位于 [jarvisvla/train/train.py](jarvisvla/train/train.py)
- 训练数据整理主要依赖 [jarvisvla/train/data_collator.py](jarvisvla/train/data_collator.py)
- 默认训练脚本只跑一套配置，位于 [scripts/train/vla_qwen2_vl_7b_sft.sh](scripts/train/vla_qwen2_vl_7b_sft.sh)
- 默认参数里已经显式冻结了视觉编码器和视觉适配器

这意味着当前仓库更像是论文里的 Stage III 主体，而不是完整的 Stage I + Stage II + Stage III。

## 二、三阶段补齐目标

### Stage I: Post-Training Language Models

目标是先提升语言模型的通用语言能力、知识与指令遵循能力，再进入多模态阶段。

### Stage II: Post-Training Vision Encoder and Language Models

目标是提升视觉表征、图文对齐、空间 grounding 和视觉问答能力，让视觉编码器和语言模型在多模态任务上更稳。

### Stage III: Action Post-Training for Interaction

目标是把前两阶段得到的视觉语言能力进一步转化成 Minecraft 中可执行的动作策略。

## 三、需要新增的配置

### 1. 阶段化训练总配置

建议新增一个统一的阶段配置文件或配置组，用来显式表达：

- 当前训练属于哪一阶段
- 本阶段使用哪些数据集
- 本阶段冻结哪些模块
- 本阶段使用什么 loss 目标
- 本阶段训练多少 steps / epoch
- 本阶段是否允许图像增强、是否启用 grounding 标注

当前 [jarvisvla/train/utils_train.py](jarvisvla/train/utils_train.py) 里的 `MoreConfig` 只有冻结开关和像素范围，建议扩展为阶段化配置，而不是只保留模块冻结参数。

### 2. 模型冻结与解冻策略配置

当前训练脚本里已经有：

- `fix_visual_encoder`
- `fix_visual_adapter`
- `fix_language_backbone`
- `fix_lm_head`

但这些只是参数级开关，不足以表达完整三阶段流程。建议补充：

- `stage_name`
- `freeze_policy`
- `trainable_modules`
- `loss_mix_weights`
- `resume_from_stage_checkpoint`

### 3. 数据集配置

每一阶段都应有独立的数据配置，至少包含：

- 数据集名称或路径
- train / valid / test 划分
- 样本 schema
- 采样比例
- 最大长度
- 图像输入约束
- 特殊标注字段

## 四、需要新增的脚本

### 1. Stage I 训练脚本

建议新增类似下面的脚本：

- `scripts/train/stage1_language_posttrain.sh`

职责：

- 启动语言模型后训练
- 使用纯文本或通用图文指令数据
- 默认冻结视觉模块，只训练语言侧需要更新的参数

### 2. Stage II 训练脚本

建议新增类似下面的脚本：

- `scripts/train/stage2_vision_language_posttrain.sh`

职责：

- 启动视觉编码器与语言模型的联合后训练
- 使用图文 QA、caption、grounding、视觉推理类数据
- 允许按配置解冻视觉编码器或适配层

### 3. Stage III 训练脚本

建议把当前脚本明确标成 Stage III，例如：

- `scripts/train/stage3_action_posttrain.sh`

职责：

- 使用 Minecraft 交互数据进行动作后训练
- 保留当前 action token / collator / SFT 流程
- 明确只负责交互策略学习

### 4. 分布式脚本扩展

如果三阶段训练都可能放大规模执行，建议为每阶段再补：

- 单机版
- 多 GPU 版
- 多机版

也就是说，脚本层至少要能覆盖：

- `stage1` 单机 / 多卡 / 多机
- `stage2` 单机 / 多卡 / 多机
- `stage3` 单机 / 多卡 / 多机

## 五、需要新增的数据

### Stage I 数据

建议补充大规模语言后训练数据，例如：

- 通用指令跟随数据
- 纯文本问答数据
- 解释、总结、推理类数据
- 可能的代码或知识型文本数据

关键点是：这一阶段不应依赖 Minecraft 动作轨迹，而应优先增强语言能力。

### Stage II 数据

建议补充多模态视觉数据，例如：

- 图文问答
- 图像描述
- 视觉 grounding
- 带 bounding box / point 的视觉理解样本
- 多轮图文对话

当前 [jarvisvla/train/data_collator.py](jarvisvla/train/data_collator.py) 已经支持 image、point、bbox，所以代码结构对这一阶段是兼容的，但还缺对应的数据源与训练配置。

### Stage III 数据

建议补充 Minecraft 交互轨迹数据，例如：

- 观测图像
- 自然语言任务指令
- 历史上下文
- 动作 token 序列
- 成功 / 失败标记
- 必要时的 GUI 状态与中间子目标

这一类数据要和 [jarvisvla/inference/action_mapping.py](jarvisvla/inference/action_mapping.py) 的动作映射一致，保证训练标签和推理动作空间对齐。

## 六、建议的数据 schema

### Stage I

- `text`
- `instruction`
- `response`
- 可选：`conversation`

### Stage II

- `text`
- `image`
- `conversation`
- 可选：`point`
- 可选：`bbox`

### Stage III

- `id`
- `conversations`
- `image` 或 `image_bytes`
- `action`
- `state`
- `history`
- 可选：`reward`
- 可选：`task_name`

## 七、需要改造的代码点

### 1. 训练入口

建议在 [jarvisvla/train/train.py](jarvisvla/train/train.py) 中加入阶段分支，例如：

- 根据 `stage_name` 选择不同数据集和 collator
- 根据阶段决定冻结策略
- 根据阶段决定是否加载上一阶段 checkpoint

### 2. 数据整理器

建议把 [jarvisvla/train/data_collator.py](jarvisvla/train/data_collator.py) 拆成更明确的阶段适配器，例如：

- language collator
- vision-language collator
- action collator

这样可以减少一个 collator 同时处理过多 schema 的复杂度。

### 3. 训练脚本

建议把当前 `scripts/train/*.sh` 按阶段重新命名和组织，避免所有脚本都指向同一个通用后训练入口。

### 4. 数据加载

当前代码使用 `load_dataset(sft_script_args.dataset_name)` 直接读一个数据集。若要完整支持三阶段，建议补充：

- 多数据集混合采样
- 不同阶段不同 split 读取
- 不同阶段不同样本权重

## 八、建议的实施顺序

### 1. 先补 Stage III 的显式化

把当前训练明确标成 Stage III，并固定好 action 数据 schema。

### 2. 再补 Stage II

先让视觉语言数据进入单独脚本与单独配置，验证冻结策略和 collator 是否稳定。

### 3. 最后补 Stage I

增加通用语言后训练的数据和脚本，然后再通过 Stage I -> Stage II -> Stage III 的 checkpoint 串联。

## 九、最小可落地版本

如果希望先做一个最小版本，不追求完全复现论文，可以按下面顺序：

1. 新增 `stage1` / `stage2` / `stage3` 三个训练脚本
2. 新增一个阶段配置文件，统一控制冻结策略和数据源
3. 给三阶段准备各自的数据集或数据视图
4. 把 Stage III 现有脚本固定为动作后训练入口
5. 逐步增加多数据集混合、分阶段 checkpoint 串联和评估

## 十、结论

当前仓库已有的是 Stage III 的主体工程能力，以及面向多模态后训练的通用训练框架。要补齐论文中的完整三阶段训练，核心不是单纯改一个训练参数，而是要把：

- 阶段配置
- 数据集组织
- 训练脚本
- 冻结策略
- checkpoint 串联

这五部分一起补起来。
