# JarvisVLA 版本化变更记录

记录项目在各个变更版本中的文件级新增、删除、修改及目的说明；并为每个版本预留安装/训练/测试操作指导区，支持后续持续追加。

## 变更版本 V1

#### 实现内容：可以跑通基于minecraft-vla-sft数据集的后训练，对应第3阶段训练的主要部分。

### A. 文件变更记录（新增/删除/修改 + 目的说明）

#### A.1 统计

- 新增文件: 14
- 删除文件: 0
- 修改文件: 13

#### A.2 文档类

| 文件 | 变更类型 | 目的说明 |
|---|---|---|
| `PROJECT_STRUCTURE.md` | 新增 | 补充项目目录结构说明，便于快速定位模块。 |
| `THREE_STAGE_TRAINING_CHECKLIST.md` | 新增 | 待实现的完整的三阶段训练流程检查清单。 |
| `jarvisvla/evaluate/README.md` | 新增 | 说明评测模块使用方式和参数。 |
| `jarvisvla/inference/README.md` | 新增 | 说明推理模块部署与调用方式。 |
| `jarvisvla/train/README.md` | 新增 | 说明训练模块入口与配置要点。 |
| `jarvisvla/utils/README.md` | 新增 | 说明工具模块职责与使用方式。 |

#### A.3 测试脚本类

| 文件 | 变更类型 | 目的说明 |
|---|---|---|
| `test/analyze_minecraft_vla_sft_dataset.py` | 新增 | 新增数据集分析脚本，用于统计和诊断 minecraft-vla-sft 数据集的样本分布。 |
| `test/inspect_minecraft_vla_samples.py` | 新增 | 新增样本巡检脚本，用于抽查 minecraft-vla-sft 数据集的样本质量与字段完整性。 |
| `scripts/inference/serve_vllm_2gpu_multi_image.sh` | 新增 | 新增多卡多图推理服务脚本，与当前环境适配（2卡4090）。 |
| `scripts/evaluate/rollout-kill.sh` | 修改 | 适配评测服务地址、并发数、最大帧数、模型路径，并新增`--allow-multi-image`参数支持（2卡4090）。 |
| `scripts/evaluate/rollout-mine.sh` | 修改 | 适配评测服务地址、模型路径，并新增`--allow-multi-image`参数支持（2卡4090）。 |
| `scripts/evaluate/rollout-gui.sh` | 修改 | 适配评测服务地址、模型路径，并新增`--allow-multi-image`参数支持（2卡4090）。 |
| `scripts/evaluate/rollout-kill2.sh` | 新增 | 新增 kill 任务评测变体脚本，用于新训练好的模型。 |
| `scripts/evaluate/rollout-mine2.sh` | 新增 | 新增 mine 任务评测变体脚本，用于新训练好的模型。 |
| `scripts/evaluate/rollout-gui2.sh` | 新增 | 新增 GUI 任务评测变体脚本，用于新训练好的模型。 |
| `scripts/train/vla_qwen2_vl_7b_sft-multi-GPU.sh` | 修改 | 适配数据集路径（改为minecraft-vla-sft）、模型路径、添加W&B上报配置与离线模式容错（x卡H800）。 |
| `scripts/train/vla_qwen2_vl_7b_sft-multi-GPU_test.sh` | 新增 | 新增训练测试脚本，用于快速验证训练配置。 |

#### A.4 实现代码类

| 文件 | 变更类型 | 目的说明 |
|---|---|---|
| `jarvisvla/evaluate/agent_wrapper.py` | 修改 | 新增`allow_multi_image`参数支持多图评测；默认单图模式以兼容更严格的vLLM部署。 |
| `jarvisvla/evaluate/evaluate.py` | 修改 | 新增`make_init_inventory_callback`统一处理MineStudio版本兼容性；重构CommandsCallback装配逻辑；新增`--allow-multi-image`命令行参数。 |
| `jarvisvla/evaluate/env_helper/craft_agent.py` | 修改 | 调整import路径：从`minestudio.models.shell.craft_agent`改为本地`jarvisvla.evaluate.env_helper.craft_agent`。 |
| `jarvisvla/evaluate/env_helper/smelt_agent.py` | 修改 | 调整import路径：从`minestudio.models.shell.gui_agent`改为本地`jarvisvla.evaluate.env_helper.gui_agent`。 |
| `jarvisvla/train/train.py` | 修改 | 新增Qwen2-VL专用chat template（`DEFAULT_QWEN2_VL_CHAT_TEMPLATE`）；支持从`chat_template.json`加载；增加模板与数据集兼容性验证；`apply_chat_template`安全渲染兜底。 |
| `jarvisvla/train/utils_train.py` | 修改 | 新增`get_effective_numel`函数，兼容ZeRO-3分片下`ds_numel`属性，解决参数计数为0的问题。 |

#### A.5 其他类

| 文件 | 变更类型 | 目的说明 |
|---|---|---|
| `.gitignore` | 修改 | 扩展忽略规则，减少日志、缓存和本地临时文件进入版本管理。 |
| `requirements.txt` | 修改 | 锁定关键依赖版本，提升环境可复现性；调整部分依赖版本约束。 |
| `setup.py` | 修改 | 用简单的逐行读取替代`pkg_resources`解析，修复兼容性问题；调整Worker的import路径。 |
| `JARVIS-VLA.pdf` | 新增 | 补充外部材料（论文或说明文档）用于背景参考。 |

### B. 安装/训练/测试指南

#### B.1 安装

首先创建conda环境：
```shell
conda create -n mcvla python=3.10
conda activate mcvla
conda install --channel=conda-forge openjdk=8 -y
```

安装torch：
```shell
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

安装主要的依赖环境：
```shell
pip install -r requirements.txt
```

安装vllm、flash-attn：
```shell
pip install vllm==0.8.2
pip install flash-attn==2.7.2.post1  # flash-attn建议到github下载编译好的文件，手动安装
```

最后运行setup.py：
```shell
python setup.py develop
```

#### B.2 训练

当前仅支持第三阶段VLA阶段的后训练，且不包含最后针对测评任务的微调。

首先下载好 minecraft-vla-sft 数据集和 Qwen2-VL-7B 模型，并确保 scripts/train 目录下的脚本`vla_qwen2_vl_7b_sft-multi-GPU_test.sh`和`vla_qwen2_vl_7b_sft-multi-GPU.sh`中的数据集和模型路径与实际一致。

可以先使用较少的显卡资源（如2卡H800）测试训练程序能否跑通。
```shell
sh scripts/train/vla_qwen2_vl_7b_sft-multi-GPU_test.sh
```

跑通后可以正式进行训练，注意设置`vla_qwen2_vl_7b_sft-multi-GPU.sh`中的显卡数与实际资源相符。
```shell
sh scripts/train/vla_qwen2_vl_7b_sft-multi-GPU.sh
```

当前开源的 minecraft-vla-sft 数据集仅为原论文训练数据的子集，约3M帧样本，根据论文描述，原训练数据至少有10~20M帧样本，实际训练出的模型效果可能并不理想。且当前数据集仅使用了单图输入、单步动作输出，原训练过程实际上涉及了多图输入、多步动作输出。

#### B.3 测试

当前的测评脚本均在2卡4090上跑通，使用其他显卡配置时需考虑修改脚本的参数。

首先启动vllm服务器。如果想对原论文提供的训练好的模型进行测评，需要下载模型 JarvisVLA-Qwen2-VL-7B，并在脚本`serve_vllm_2gpu_multi_image.sh`中设置正确的模型路径；如果想对自己训练的模型进行测评，也需要将模型路径设置为自己训练的输出目录。
```shell
sh scripts/inference/serve_vllm_2gpu_multi_image.sh
```

对论文提供的模型进行测评，分别运行如下任务测评脚本。
```shell
sh scripts/evaluate/rollout-kill.sh
sh scripts/evaluate/rollout-mine.sh
sh scripts/evaluate/rollout-gui.sh
```

对自己训练的模型进行测评，分别运行如下任务测评脚本。
```shell
sh scripts/evaluate/rollout-kill2.sh
sh scripts/evaluate/rollout-mine2.sh
sh scripts/evaluate/rollout-gui2.sh
```

## 后续版本追加模板

后续新增版本请按以下结构追加:

```markdown
## 变更版本 V2

### A. 文件变更记录（新增/删除/修改 + 目的说明）

### B. 安装/训练/测试指南
```