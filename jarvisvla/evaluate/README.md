# jarvisvla/evaluate 目录说明

本目录负责评测执行：在 Minecraft 环境中调用模型进行 rollout，并统计任务成功率与过程数据。

## 文件功能

- `__init__.py`
  - 包初始化文件。

- `evaluate.py`
  - 评测主入口。
  - 读取任务配置（Hydra）。
  - 初始化环境与回调。
  - 调用 agent 逐步执行动作，记录奖励与成功状态。
  - 支持单 worker 与基于 Ray 的并行评测。

- `agent_wrapper.py`
  - 模型代理封装。
  - 负责构造指令、历史上下文、recipe 提示等。
  - 调用 vLLM(OpenAI 兼容接口) 获取输出。
  - 将输出解析为动作序列返回给环境。

- `draw_utils.py`
  - 评测结果可视化。
  - 绘制成功率曲线、loss 曲线、推理步数对比等图像。

- `assets/`
  - 评测相关资源。
  - 包含指令模板与 recipes 等静态数据。

- `config/`
  - 任务 YAML 配置。
  - 按任务类别组织（craft/kill/mine/smelt），定义任务文本、奖励、初始背包等。

- `env_helper/`
  - 环境 GUI 操作辅助层。
  - 提供工作台/熔炉打开、背包拖拽、合成/熔炼等细粒度操作逻辑。

## 评测链路

1. `evaluate.py` 读取任务配置并初始化环境。
2. `agent_wrapper.py` 组织 prompt 并请求模型。
3. 模型输出被解码为动作后执行到环境。
4. 成功率与过程记录由 `draw_utils.py` 等模块统计展示。
