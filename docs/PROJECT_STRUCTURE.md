# JarvisVLA 项目结构说明

本文档用于快速理解仓库整体结构与模块关系，侧重“训练-推理-评测”主流程。

## 1. 项目定位

JarvisVLA 是一个面向 Minecraft 任务的视觉-语言-动作（VLA）研究项目。核心目标是让模型根据视觉观测与文字指令，输出可执行的键鼠动作序列，完成如 craft、smelt、mine、kill 等任务。

## 2. 顶层目录说明

- `jarvisvla/`：核心 Python 包，包含训练、推理、评测和工具模块。
- `scripts/`：训练、推理服务、评测启动脚本。
- `configs/`：DeepSpeed 训练配置（ZeRO stage 等）。
- `assets/`：全局资源（如特殊 token 配置）。
- `logs/`：评测日志、视频和统计结果输出目录。
- `requirements.txt`：主要依赖列表。
- `setup.py`：可编辑安装入口（`pip install -e .`）。
- `JARVIS-VLA.pdf`：对应论文文件。

## 3. `jarvisvla` 主包结构

- `jarvisvla/train/`：训练入口、数据整理（collator）、训练辅助。
- `jarvisvla/inference/`：推理侧输入封装、动作 token 映射、模型打包辅助。
- `jarvisvla/evaluate/`：环境执行与评测主循环，含任务配置和 GUI 操作辅助。
- `jarvisvla/utils/`：通用工具函数（当前主要是 JSON 文件读写）。

## 4. 端到端流程

1. 在 `train/` 中进行 SFT 训练，得到 checkpoint。
2. 使用 `scripts/inference/serve_vllm.sh` 启动模型服务。
3. 使用 `scripts/evaluate/*.sh` 在 Minecraft 环境上 rollout。
4. 在 `logs/` 中查看视频、成功率与统计结果。

## 5. 文档导航

为了便于逐层阅读，本仓库已在下列子目录内提供细化说明文档：

- `jarvisvla/train/README.md`
- `jarvisvla/inference/README.md`
- `jarvisvla/evaluate/README.md`
- `jarvisvla/utils/README.md`
