# JarvisVLA 三阶段训练检查清单

本文档根据当前仓库代码、公开数据集结构，以及已完成的 Stage I / II / III 数据分析结果重写，目标是把“论文中的三阶段方法”与“当前开源仓库里真实可用的能力”对齐。

核心结论先放前面：

- Stage I 的公开数据基本齐，主要是 `minecraft-vlp` 里的 world knowledge QA。
- Stage II 的图文对齐数据基本齐，但 grounding 明显是 point-centric，公开包里没有 native bbox-answer。
- Stage III 的动作轨迹数据已经形成独立数据集 `minecraft-vla-sft`，但当前训练入口更像一个通用 SFT 流程，而不是严格的三阶段管线。

## 1. 当前仓库里已经有什么

当前代码不是严格分阶段实现，而是一个通用的多模态 SFT 框架：

- 训练入口在 [jarvisvla/train/train.py](../jarvisvla/train/train.py)
- 数据整理主要在 [jarvisvla/train/data_collator.py](../jarvisvla/train/data_collator.py)
- 训练配置扩展在 [jarvisvla/train/utils_train.py](../jarvisvla/train/utils_train.py)
- 动作映射在 [jarvisvla/inference/action_mapping.py](../jarvisvla/inference/action_mapping.py)
- 默认训练脚本仍是单一路径的 post-training / SFT 入口

从代码形态看，它更接近 Stage III 主体工程，再加上一部分面向 Stage II 的通用 collator 支撑，而不是一个已经完整拆分好的 Stage I + Stage II + Stage III 系统。

## 2. 公开数据的真实覆盖

### 2.1 `minecraft-vlp`

数据位置：`/share/public_datasets/VLA/nitrogen/minecraft-vlp`

公开包里可见的主要文件：

- `mc-qa-250312.jsonl`
- `mc-vqa-241102.jsonl`
- `mc-caption-241104.jsonl`
- `mc-grounding-point-gui.jsonl`
- `mc-grounding-point-embodied.jsonl`
- `mc-grounding-point-embodied-image5.jsonl`
- `mc-knowledge-valid.jsonl`
- `hallucination.jsonl`

对应图像资源位于 `images/` 下的多个 zip 中。

分析后的判断是：

- `mc-qa-250312.jsonl` 对应 Stage I，规模约 275,905。
- `mc-caption-241104.jsonl` 和 `mc-vqa-241102.jsonl` 对应 Stage II 的图文对齐部分，合计接近论文口径。
- `mc-grounding-*` 系列对应 Stage II 的 grounding 部分，但总量比论文的 404K 口径少很多。
- `mc-knowledge-valid.jsonl` 更像一个混合验证集，不是独立训练主集。
- `hallucination.jsonl` 更像缺图控制样本或负样本。

### 2.2 `minecraft-vla-sft`

数据位置：`/share/public_datasets/VLA/nitrogen/minecraft-vla-sft`

这是明显的 Stage III 轨迹数据集，结构是：

- user 提任务和 observation image
- assistant 输出动作 token 序列

它的训练目标是 action post-training，而不是图文问答。

## 3. 代码里已经支持到哪一步

### 3.1 已有的通用能力

`train.py` 目前是一个通用训练入口：

- 直接用 `load_dataset(...)` 读数据
- 统一走 SFT / chat template 流程
- 没有显式 Stage I / Stage II / Stage III 分支
- 没有内建“按阶段切换数据源和冻结策略”的总控逻辑

`utils_train.py` 里已经有一些冻结相关配置，但还是通用参数，不是阶段化策略。

### 3.2 Grounding 处理现状

`data_collator.py` 已经有 `point` 和 `bbox` 两条分支，说明工程上意识到了空间 grounding 这件事：

- `point` 分支处理 point 监督
- `bbox` 分支预留了 bbox 处理路径

但结合数据分析结果来看，公开 `minecraft-vlp` 的真实监督是 point-centric 的：

- 全局回答类型里 `point` 占主导
- `bbox` answer 没有形成公开数据里的 native supervision
- 显式 bbox prompt 只出现在少量 `hallucination.jsonl` 样本中，而且回答也不是 bbox

所以更准确的判断是：

- 代码里有 bbox 路径的骨架
- 数据里没有可直接复用的 bbox-answer 主集
- 如果要做 bbox 扩增，需要自己构造派生数据集

## 4. 三阶段训练的推荐落点

### Stage I: 语言后训练

目标：先稳住语言能力、知识问答能力和指令遵循能力。

建议数据：

- `minecraft-vlp/mc-qa-250312.jsonl`
- 可选：`mc-knowledge-valid.jsonl` 里的无图样本做验证

建议特征：

- 冻结视觉分支
- 以纯文本或无图问答为主
- 以稳定 chat template 和语言建模为核心

### Stage II: 视觉语言对齐 + grounding

目标：提升图文对齐、VQA、caption、空间定位能力。

建议数据：

- `minecraft-vlp/mc-caption-241104.jsonl`
- `minecraft-vlp/mc-vqa-241102.jsonl`
- `minecraft-vlp/mc-grounding-point-gui.jsonl`
- `minecraft-vlp/mc-grounding-point-embodied.jsonl`
- `minecraft-vlp/mc-grounding-point-embodied-image5.jsonl`

可选辅助：

- `mc-knowledge-valid.jsonl` 里的有图样本
- `hallucination.jsonl` 作为低权重负样本或控制样本

这一阶段最重要的工程点是：

- 要明确区分 caption / VQA / grounding 的采样比例
- 要保留多图历史样本，尤其是 `embodied-image5`
- 如果要扩展 bbox，应该把 point 样本重写成 bbox 监督，作为派生数据

### Stage III: 动作后训练

目标：把视觉语言能力转成 Minecraft 交互策略。

建议数据：

- `minecraft-vla-sft`

这一阶段的数据和前两阶段不是同一类 schema，应该单独看待，不能和图文对齐任务混跑成同一条数据流。

## 5. 如果要做 bbox 增广，应该怎么定义

当前分析已经确认：公开数据里没有足够的 native bbox-answer，但 `source.bbox` 是存在的，所以可以做派生数据。

推荐定义是：

- 输入端仍然是原始图像和 grounding prompt
- 将“回答 point”的 prompt 重写为“回答 bbox”的 prompt
- 标签从 point 转成 bbox
- 作为一个新的 Stage II augmentation 子集，不要当成原始公开数据本身

这个方案的意义是扩充监督形式，而不是证明公开包本来就有 bbox 标注训练集。

## 6. 建议补齐的工程项

### 6.1 阶段配置

建议新增明确的阶段配置，而不是只靠几个冻结开关：

- `stage_name`
- `dataset_list`
- `freeze_policy`
- `trainable_modules`
- `loss_mix_weights`
- `resume_from_stage_checkpoint`

### 6.2 训练脚本

建议把训练脚本显式拆成三份：

- Stage I language post-train
- Stage II vision-language post-train
- Stage III action post-train

### 6.3 数据加载

建议按数据子集显式分支，不要把 `minecraft-vlp` 当成统一 schema：

- QA / VQA / caption / grounding 应该分开采样
- `mc-knowledge-valid` 应该作为混合验证集合处理
- `hallucination` 应该作为辅助控制集合处理

### 6.4 bbox 派生数据生成

如果后续要真做 bbox 扩增，建议单独加一个离线生成脚本，把 point 样本转成 bbox 样本，再喂给 Stage II collator。

## 7. 推荐实施顺序

1. 先把当前训练入口明确标成 Stage III。
2. 再把 Stage II 数据子集和 grounding 处理独立出来。
3. 最后补 Stage I 的纯语言后训练数据和脚本。
4. 如果需要 bbox 扩增，再加离线派生数据生成器。

## 8. 最小可落地版本

如果先追求能跑通而不是完全复现论文，最低限度建议做这四件事：

1. 把 `minecraft-vlp` 和 `minecraft-vla-sft` 的用途分开。
2. 把 Stage II 的 point grounding 和 bbox augmentation 分开处理。
3. 给 `mc-knowledge-valid` 和 `hallucination` 明确辅助定位，而不是主训练定位。
4. 给 Stage I / II / III 三阶段各自准备独立的训练脚本与配置。

## 9. 结论

当前仓库不是一个已经完整落地的三阶段训练系统，但它已经具备了足够清楚的骨架：

- Stage I 的公开数据基本可用。
- Stage II 的结构基本可用，但 grounding 明显 point-centric，bbox 需要派生生成。
- Stage III 的数据已经成型，但训练入口仍需显式阶段化。

下一步最值得做的，不是继续猜数据，而是把阶段配置、数据源、collator 逻辑和 bbox 派生流程正式工程化。
