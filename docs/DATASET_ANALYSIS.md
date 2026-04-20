# JarvisVLA 数据集分析报告

本文基于以下四个脚本的运行结果，并结合论文 [JARVIS-VLA.pdf](JARVIS-VLA.pdf) 中的三阶段训练描述，对当前公开数据集的构成、分布以及对完整三阶段训练的覆盖程度做一次系统分析：

- [test/analyze_minecraft_vlp_dataset.py](../test/analyze_minecraft_vlp_dataset.py)
- [test/inspect_minecraft_vlp_samples.py](../test/inspect_minecraft_vlp_samples.py)
- [test/analyze_minecraft_vla_sft_dataset.py](../test/analyze_minecraft_vla_sft_dataset.py)
- [test/inspect_minecraft_vla_sft_samples.py](../test/inspect_minecraft_vla_sft_samples.py)

对应的运行结果文件为：

- [test/logs/minecraft_vlp_report_20260420_145726.json](../test/logs/minecraft_vlp_report_20260420_145726.json)
- [test/logs/vlp_samples.json](../test/logs/vlp_samples.json)
- [test/logs/label_hierarchy_with_instructions_20260420_150536.json](../test/logs/label_hierarchy_with_instructions_20260420_150536.json)
- [test/logs/vla_sample.json](../test/logs/vla_sample.json)

## 1. 结论先行

公开数据的整体形态和论文的三阶段方法是对得上的，但完整性并不均衡。

第一阶段相关的 world knowledge 数据集规模与论文报告非常接近；第二阶段相关的 caption、VQA 和部分 grounding 数据也能对上论文中的任务类型，但 grounding 总量明显偏少；第三阶段的 minecraft-vla-sft 是一个典型的动作轨迹后训练集，结构完整，但它只是论文中 action post-training 的一部分，而且已经证实是原始训练数据的子集。

因此，从当前公开数据看，更合理的判断是：

1. Stage I 数据基本可用，规模接近论文口径。
2. Stage II 的视觉-语言对齐数据基本可用，但 grounding 侧很可能不完整。
3. Stage III 的动作轨迹数据是单独的一条主线，工程上已经形成了清晰可训练的 SFT 数据集，但和论文的全量训练数据仍存在明显差距。

## 2. 数据集总体结构

### 2.1 minecraft-vlp

`minecraft-vlp` 的当前目录结构是一个由多个 JSONL 文件和多个图像压缩包组成的多任务数据集合，而不是单一 HF split。

顶层文件共有 8 个 JSONL：

| 文件 | 记录数 | 主要任务类型 |
|---|---:|---|
| hallucination.jsonl | 196 | 无图问答 / grounding 负样本 |
| mc-caption-241104.jsonl | 9,023 | 图像描述 |
| mc-grounding-point-embodied-image5.jsonl | 2,439 | 5 帧历史图像的 embodied grounding |
| mc-grounding-point-embodied.jsonl | 29,710 | embodied grounding |
| mc-grounding-point-gui.jsonl | 195,515 | GUI grounding |
| mc-knowledge-valid.jsonl | 50 | 小规模混合验证集 |
| mc-qa-250312.jsonl | 275,905 | Minecraft world knowledge QA |
| mc-vqa-241102.jsonl | 24,774 | Minecraft VQA |

全量记录数为 537,612。

对应的 image 资源放在 `images/` 下，且以同名 zip 归档：

- mc-caption-241104.zip
- mc-grounding-point-embodied-image5.zip
- mc-grounding-point-embodied.zip
- mc-grounding-point-gui.zip
- mc-vqa-241102.zip

补充说明：`mc-knowledge-valid.jsonl` 和 `hallucination.jsonl` 没有单独对应的 zip 包，但 `mc-knowledge-valid.jsonl` 的 40 条图片引用可以全部在上述 4 个任务 zip 中找到，说明它更像是从其他子集中抽样出来的混合验证集，而不是独立图片包。

### 2.2 总体类别分布

按一层标签统计，`minecraft-vlp` 的数据量分布如下：

| 一级标签 | 数量 | 占比 |
|---|---:|---:|
| qa | 275,915 | 51.32% |
| gui | 195,525 | 36.37% |
| embody | 32,159 | 5.98% |
| vqa | 24,773 | 4.61% |
| caption | 9,033 | 1.68% |
| grounding | 196 | 0.04% |
| recipe_book | 11 | 0.00% |

这说明数据集合的主体是三块：

1. 语言知识 QA。
2. GUI / embodied 的空间 grounding。
3. 图文描述与 VQA。

### 2.3 和论文口径的对照

论文中给出的 Stage I / Stage II 非轨迹数据规模大致是：

- world knowledge: 约 277K
- visual-language alignment: 约 35K
- spatial grounding: 约 404K

当前公开 `minecraft-vlp` 的可见总量与这个口径并不完全一致：

| 任务块 | 论文规模 | 当前公开数据可见规模 | 观察 |
|---|---:|---:|---|
| Stage I world knowledge | 约 277K | 275,905 | 基本一致 |
| Stage II alignment | 约 35K | 33,797 到 33,847 | 基本接近 |
| Stage II grounding | 约 404K | 227,664 | 明显偏少 |

其中 alignment 的 33,797 只统计 caption + VQA；若把 50 条 `mc-knowledge-valid` 也算进来，则约 33,847。无论哪种口径，都接近论文的 35K。

真正偏离最大的是 grounding：当前可见的 GUI + embodied + embodied-image5 合计只有 227,664，距离论文的 404K 还差约 176K 左右。这是一个很重要的信号，说明当前公开包很可能不是论文 Stage II grounding 数据的全量版。

## 3. minecraft-vlp 各子集分析

### 3.1 mc-qa-250312.jsonl

这是最接近 Stage I 的核心数据。

- 记录数：275,905
- 一级标签：qa
- 二级标签：wiki
- 三级标签：self-instruct
- 对话轮次：几乎全部是 3 轮，少量是 5 轮
- 图片：完全没有图像
- 任务形态：system + user + assistant 的纯文本问答

这部分数据和论文中的 world knowledge stage 对应得非常直接。它的结构非常标准，适合语言后训练：

1. 输入侧没有视觉依赖。
2. 输出侧是自然语言回答。
3. 对话模板稳定。

从样本看，它更像是 Minecraft 语境下的知识问答与指令理解集合，而不是动作轨迹数据。

### 3.2 mc-caption-241104.jsonl

这是典型的 Stage II 视觉-语言对齐数据。

- 记录数：9,023
- 一级标签：caption
- 二级标签：contractor
- 三级标签：question_based / prompt_based
- 图片：每条 1 张
- 对话结构：2 轮，user 提问，assistant 描述图像
- 图像压缩包：存在，且图片数与记录数一致

这类样本的作用是让模型学习 Minecraft 场景的图像描述能力。它和论文里“captioning”这一类对齐任务完全一致。

### 3.3 mc-vqa-241102.jsonl

这是 Stage II 里最接近 VQA 的数据。

- 记录数：24,774
- 一级标签：vqa
- 二级标签：contractor
- 三级标签：大部分是 Minecraft，另有少量 gpt-4o 风格样本
- 对话轮次：分布较广，常见 6 到 16 轮，也有更长的 18、20、22 轮
- 图片：每条 1 张
- 图像压缩包：存在，且主要成员与记录对得上

这个子集有两个值得注意的特点：

1. 它不是单轮 VQA，而是有多轮追问和上下文延续，说明它不只是简单问答，更像是“围绕截图进行连续视觉推理”的训练样本。
2. 其中有少量 `recipe_book` 相关样本，说明这个子集不是纯粹的单一视觉问答，而是混入了少量文档式或手册式推理样本。

### 3.4 mc-grounding-point-gui.jsonl

这是 Stage II spatial grounding 的核心部分之一，也是当前 VLP 中规模最大的视觉 grounding 文件。

- 记录数：195,515
- 一级标签：gui
- 二级标签：Minecraft
- 三级标签：point
- 图片：每条 1 张
- 对话结构：2 轮，user 请求 point，assistant 回 point
- 结构特征：source 中有 bbox、image_url、points 等字段

它的语义很明确：给定 GUI 截图，要求模型定位一个 UI 元素或物品位置，输出点坐标。

从 source 分布看，`label`、`image_url`、`points`、`bbox` 都出现得很稳定，这说明数据本身是为了空间定位监督而构造的。

但有一个关键现象：zip 里的图片数 211,961，大于 195,515 条记录。这意味着压缩包里可能存在额外图片、重复图片或预留样本。也就是说，记录数和磁盘图片数不是严格一一对应。

### 3.5 mc-grounding-point-embodied.jsonl

这是 Stage II 的 embodied grounding 样本。

- 记录数：29,710
- 一级标签：embody
- 二级标签：Minecraft
- 三级标签：point
- 图片：每条 1 张
- source 里包含 video_path、points、bbox、image_url、label

这部分和 GUI grounding 的区别是它来自更偏“世界观察”的 embodied 场景，不是纯 GUI。它更接近论文里所说的 spatial grounding 和视觉识别能力增强。

和 GUI grounding 一样，这里的 zip 也不是严格一一对应：记录数 29,710，而图片成员数 29,708。

### 3.6 mc-grounding-point-embodied-image5.jsonl

这是整个 `minecraft-vlp` 里最能体现论文“多图历史上下文”的子集。

- 记录数：2,439
- 一级标签：embody
- 二级标签：Minecraft
- 三级标签：point
- 图片：每条样本不是 1 张，而是 2 到 5 张
- 对话结构：常见 10 轮，多轮历史图像和 point 监督交替出现
- source 里有 image_urls、video_path、points、bbox、label、action

这个子集非常重要，因为它直接体现了论文里说的 non-Markovian 设计，也就是把历史观察图像放进 prompt。它对后续想实现的完整三阶段系统很关键：

1. 说明模型并不只依赖单帧输入。
2. 说明 Stage II 的数据已经向时序感知和空间 grounding 过渡。
3. 说明如果要复现论文方法，不能只保留单图 QA / caption。

### 3.7 mc-knowledge-valid.jsonl

这个子集更像一个混合验证/诊断集合，而不是纯训练集。

- 记录数：50
- 一级标签分布：caption / embody / gui / qa / vqa 各 10 条
- 对话轮次：2、3、4、6、8、10、12、14、16、20、22 都有少量
- 图片：40 条有图，10 条无图
- 图片引用追踪结果：40 条图片全部可在 `mc-grounding-point-gui.zip`、`mc-grounding-point-embodied.zip`、`mc-caption-241104.zip`、`mc-vqa-241102.zip` 这 4 个压缩包中定位到，每个压缩包命中 10 条
- 当前目录里没有单独对应的 zip 归档

从用途上看，这 50 条样本可以拆成两部分理解：

1. 无图样本更适合用于 Stage I 的语言理解和知识问答验证。
2. 有图样本更适合用于 Stage II 的视觉语言对齐或 grounding 验证。

它的作用更像是：

1. 提供一个小规模、跨任务的验证视图。
2. 用来检查模型在不同任务形态上的泛化。
3. 作为对齐质量的 sanity check，而不是主要训练语料。

当前公开目录下没有为它单独放一个 images 压缩包，这一点值得后续确认。可能是存放路径不同，也可能是公开包未完整收录。

### 3.8 hallucination.jsonl

这是一个很小但很有语义价值的子集。

- 记录数：196
- 一级标签：grounding
- 对话：2 轮
- 图片：0
- 典型输出：assistant 直接回答 “An image is needed to provide an answer.”

这类样本更像负样本或控制样本，提示模型在缺图时不要胡乱生成 grounding 结果。它对于训练或评估 hallucination 抑制是有意义的，但不应被当作主训练数据。

## 4. minecraft-vlp 的包装与完整性判断

从脚本结果看，`minecraft-vlp` 的包装方式是清晰的，但数据完整性并不均匀。

### 4.1 整体上是成体系的

它至少覆盖了论文 Stage I 和 Stage II 所需的主要任务类型：

- 世界知识 QA
- 图像描述
- VQA
- GUI grounding
- embodied grounding
- 多图时序 grounding

### 4.2 但 Stage II grounding 明显偏少

这是最强的异常信号。

论文说 spatial grounding 约 404K，而当前公开可见的 grounding 相关样本只有 227,664 左右。这个缺口足以说明两种可能：

1. 公开数据是裁剪版或子集。
2. 论文统计口径包含了当前目录外的额外数据源。

无论是哪一种，都意味着：如果你后续要做完整三阶段训练，不能把当前 `minecraft-vlp` 当作论文 Stage II 的全量数据。

### 4.3 文件包装存在轻微不对称

有几个值得注意的点：

1. 一些压缩包的图片成员数比 JSONL 记录数多，例如 `mc-grounding-point-gui.zip`。
2. `mc-knowledge-valid.jsonl` 没有独立压缩包，但它的图片引用可以完全从其它 4 个 zip 里追溯出来，因此它更像共享图像资源的混合验证集。
3. `hallucination.jsonl` 也没有对应图像包，而且样本本身就是无图控制样本。

这些都不一定是错误，但足以说明需要在数据加载层显式处理不同子集，而不能假设所有子集都遵循同一套 schema。

### 4.4 关于“把 point 样本改成 bbox 作为数据增广”的检验

你的设想是合理的，但它应当被理解为**synthetic bbox augmentation**，而不是“原始数据本来就有 bbox 监督”。

当前公开 VLP grounding 相关样本的真实监督答案形式是 point-centric 的：

- `point` 回答：266,435
- `text` 回答：480
- `bbox` 回答：0

同时，严格按 prompt 文本分类后，显式要求 `bbox` 的样本只有 64 条，而且它们全部来自 `hallucination.jsonl`，并没有形成可直接复用的 bbox supervision：

- `prompt_class=bbox` 的样本：64
- 这些样本的回答类型：全部是 `text`

这意味着：

1. 现有公开样本里没有 native bbox-answer，可以直接拿来当 bbox 监督。
2. 但你完全可以基于现有 `source.bbox` / `source.points` 做 prompt rewrite，把原来的 point 问题改写成 bbox 问题，再用 bbox 作为新答案，生成一个新的派生数据集。
3. 如果对所有 point-grounding 样本都做一次 bbox 版本增广，理论上会新增约 227,664 条记录，整体会从 227,664 变成约 455,328 条；这会超过论文里 404K 的口径，所以它不是“恰好补齐”，而是一个偏大的增广方案。
4. 如果只对那 64 条显式 bbox 问句做 bbox 派生，新增规模又太小，几乎可以忽略不计。

因此，更合理的说法是：论文里的 404K spatial grounding 规模**不能**由当前公开包自动推出，但可以通过基于 `source.bbox` 的 prompt rewrite 构造一个新的 bbox grounding 子集，作为 Stage II 的补充增强数据。这个新子集是否能恰好对齐 404K，取决于你对哪些样本做过滤、是否保留多点样本、以及是否把多图历史样本按 frame 级或 sample 级展开。

## 5. minecraft-vla-sft 的结构和 Stage III 特征

第三阶段数据集 `minecraft-vla-sft` 明确体现的是动作后训练，而不是视觉语言理解。

### 5.1 总规模与结构

- train: 3,776,475
- valid: 1,000

所有样本都具备统一的轨迹式结构，典型形式是：

1. user 输入任务指令 + 当前 observation image。
2. assistant 输出 action token 序列。

### 5.2 标签层级

train 集的一级标签分布：

| 一级标签 | 数量 | 占比 |
|---|---:|---:|
| mine_block | 2,551,748 | 67.57% |
| craft_item | 723,303 | 19.15% |
| kill_entity | 273,235 | 7.24% |
| drop | 226,766 | 6.00% |

valid 集的分布形态相近。

这说明第三阶段虽然规模很大，但任务分布很不均衡，明显偏向资源采集和少数高频动作。它不是一个“完整 Minecraft 行为空间”的均匀覆盖集，而是一个面向主要任务子域的动作后训练数据集。

### 5.3 主要 minor 类别

`mine_block` 下最常见的 minor 类别包括：

- sugar_cane
- hay_block
- oak_leaves
- grass_block
- sand
- spruce_leaves
- wheat
- oak_log
- grass
- dark_oak_log

这说明 Stage III 的动作训练重点还是在常见资源、植物、木材、矿物和 GUI 交互相关目标上。

### 5.4 样本形态

从 `vla_sample.json` 可见，典型样本具有以下特征：

- 只有 2 轮对话
- user 中包含自然语言任务说明和 observation 图像
- assistant 输出 reserved special tokens，对应动作 chunk
- 标签形态为 `trajectory | RT2 | action:target | h=0 | c=1`

这和 `minecraft-vlp` 的图文 QA / grounding 任务是明显不同的：

1. Stage III 的输出空间是动作 token，而不是自然语言答案。
2. 训练目标是 imitation learning，不是纯 SFT 文本生成。
3. 数据中的图像是即时观测，不是 caption / QA / point 的监督对象。

### 5.5 质量与异常

train 集中有 1,423 条第三标签为空，全部是空 raw third label，占比约 0.0377%。这不算严重，但说明仍有少量标签不完整的样本需要在训练前做过滤或容错。

valid 集没有这类问题。

## 6. 对完整三阶段训练的工程含义

结合论文和当前两个数据集的分析，完整三阶段训练应当这样理解：

### Stage I: Language post-training

最适合使用：

- mc-qa-250312.jsonl

可能可补充：

- mc-knowledge-valid.jsonl 中的无图样本，作为验证或小规模混合检查集

不建议直接用：

- hallucination.jsonl 作为主训练数据，它更像控制样本

### Stage II: Vision-language alignment + grounding

最适合使用：

- mc-caption-241104.jsonl
- mc-vqa-241102.jsonl
- mc-grounding-point-gui.jsonl
- mc-grounding-point-embodied.jsonl
- mc-grounding-point-embodied-image5.jsonl

可作为辅助样本：

- mc-knowledge-valid.jsonl 中的有图样本，用于小规模 Stage II 验证或补充评测
- hallucination.jsonl 作为低权重的负样本或控制样本，帮助模型在缺图或不确定输入时减少胡乱 grounding

这个阶段最需要注意的是：

1. 训练任务混合后，需要显式区分 caption/VQA/grounding 的 loss 或 sampling 策略。
2. 多图历史输入是论文的关键特征，不要只保留单图版本。
3. 当前公开包中的 grounding 总量看起来不足，可能需要补数据或补另一份未公开子集。
4. 若引入 `hallucination.jsonl`，建议只作为小比例辅助项，不要和主 grounding 数据等权混合。
5. 可以把现有 point 样本改写成 bbox 监督做增广，但它是新构造的数据，不是原始公开监督；最终规模是否接近 404K 取决于过滤和展开策略。

### Stage III: Action post-training

最适合使用：

- minecraft-vla-sft

它已经体现出动作 token 体系、历史 observation、以及任务到 action 的对齐方式，但它只是 Stage III 的当前实现版本，不代表论文中全部轨迹数据。

## 7. 对后续实现的建议

1. 数据加载层要按子集名做显式分支，不要假设 `minecraft-vlp` 是统一 schema。
2. Stage I、Stage II、Stage III 应该分别维护数据清单和采样比例。
3. Stage II 要单独处理多图历史样本，尤其是 embodied-image5 这类样本。
4. `mc-knowledge-valid` 虽然没有独立 zip，但其图片可以在其它任务压缩包中完整找到，说明它是共享图像资源的混合验证集；其中无图样本更适合 Stage I，有图样本更适合 Stage II。
5. `hallucination` 可以作为 Stage II 的小比例辅助控制样本，但不建议作为主训练数据或与主 grounding 数据等权混合。
6. 如果目标是复现论文，不应把现有 `minecraft-vlp` 当成 Stage II 全量数据，而应把它视为一个已经可用但明显不完整的公开版本。

## 8. 总结

当前公开数据已经足以支撑一个清晰的三阶段工程骨架：

- Stage I: 语言知识后训练，数据基本齐。
- Stage II: 视觉语言对齐和 grounding，形态齐，但 grounding 总量明显偏少。
- Stage III: 动作后训练，数据结构已经明确，但仍是原始训练集的子集版本。

换句话说，数据的“方向”是完整的，数据的“规模”却并不完整。对于你后续要做的二次开发，这意味着最重要的工作不是简单跑通一个训练脚本，而是先把阶段、数据源、schema 和采样策略重新梳理成一套显式的工程配置。