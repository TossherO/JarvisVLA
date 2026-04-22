'''
Author: Muyao 2350076251@qq.com
Date: 2025-03-04 23:35:08
LastEditors: Muyao 2350076251@qq.com
LastEditTime: 2025-05-28 23:10:17
'''
import logging
import os
import sys
import pathlib
from dataclasses import fields, is_dataclass

TRL_USE_RICH = os.getenv("TRL_USE_RICH", False)

from trl.scripts import init_zero_verbose, ScriptArguments, TrlParser
from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTConfig,
    get_quantization_config,
    get_kbit_device_map,
)

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"
    from rich.console import Console
    from rich.logging import RichHandler
if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)
    
from datasets import load_dataset

import torch
from tqdm.rich import tqdm
from transformers import Qwen2VLProcessor
from transformers import Qwen2VLForConditionalGeneration
from transformers import Trainer

import json
import re
from typing import Any, Dict

from rich import print
from jarvisvla.train.utils_train import (
    print_trainable_parameters,
    seed_everything,
    MoreConfig,
)
from jarvisvla import QWEN_SPECIAL_TOKENS
from jarvisvla.train.data_collator import make_collator

tqdm.pandas()    

STAGE_NAMESPACE_KEYS = [
    "script_arguments",
    "script_args",
    "training_arguments",
    "training_args",
    "model_arguments",
    "model_args",
    "more_arguments",
    "more_args",
]

DEFAULT_ROLE_AWARE_QWEN2_VL_CHAT_TEMPLATE = (
    "{% set image_count = namespace(value=0) %}"
    "{% set video_count = namespace(value=0) %}"
    "{% for message in messages %}"
    "{% if loop.first and message['role'] != 'system' %}"
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "{% endif %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n"
    "{% else %}"
    "{% for content in message['content'] %}"
    "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
    "{% set image_count.value = image_count.value + 1 %}"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif content['type'] == 'video' or 'video' in content %}"
    "{% set video_count.value = video_count.value + 1 %}"
    "<|vision_start|><|video_pad|><|vision_end|>"
    "{% elif 'text' in content %}{{ content['text'] }}{% endif %}"
    "{% endfor %}<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def _load_chat_template_from_json(model_path: str):
    template_path = pathlib.Path(model_path) / "chat_template.json"
    if not template_path.exists():
        return None
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload.get("chat_template")
        if isinstance(payload, str):
            return payload
    except Exception:
        return None
    return None


def _is_role_aware_chat_template(template: str) -> bool:
    if not template:
        return False
    return ("message['role']" in template) or ('message["role"]' in template)


def _safe_render_chat_template(tokenizer, conversations) -> str:
    try:
        rendered = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
        return rendered if isinstance(rendered, str) else ""
    except Exception:
        return ""


def _load_stage_config(config_path: str) -> Dict[str, Any]:
    config_file = pathlib.Path(config_path)
    if not config_file.exists():
        raise ValueError(f"Stage config file not found: {config_path}")
    with open(config_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Stage config must be a JSON object.")
    return payload


def _extract_stage_config_path(argv):
    for i, token in enumerate(argv):
        if token == "--stage_config_path" and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith("--stage_config_path="):
            return token.split("=", 1)[1]
    return ""


def _normalize_cli_key(token: str) -> str:
    if not token.startswith("--"):
        return ""
    key = token[2:]
    if not key:
        return ""
    if "=" in key:
        key = key.split("=", 1)[0]
    return key.replace("-", "_")


def _collect_cli_keys(argv):
    keys = set()
    for token in argv:
        normalized = _normalize_cli_key(token)
        if normalized:
            keys.add(normalized)
    return keys


def _collect_known_parser_fields():
    known = set()
    for cls in (ScriptArguments, SFTConfig, ModelConfig, MoreConfig):
        if is_dataclass(cls):
            known.update(f.name for f in fields(cls))
    return known


def _stringify_cli_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _build_stage_default_cli_args(stage_cfg: Dict[str, Any], known_fields, existing_cli_keys):
    cli_args = []
    for namespace in STAGE_NAMESPACE_KEYS:
        overrides = stage_cfg.get(namespace)
        if not isinstance(overrides, dict):
            continue
        for key, value in overrides.items():
            normalized_key = key.replace("-", "_")
            if normalized_key not in known_fields or normalized_key in existing_cli_keys:
                continue
            if value is None:
                continue
            cli_args.extend([f"--{normalized_key}", _stringify_cli_value(value)])
            existing_cli_keys.add(normalized_key)
    return cli_args


def _parse_args_with_stage_defaults():
    """解析训练参数，并在 parser 之前注入 stage 配置默认值。

    设计目标：
    1) 让 required 参数（例如 dataset_name）在参数解析阶段就可见。
    2) 保留 CLI 覆盖优先级：用户显式传参 > stage JSON 默认值。
    """
    raw_argv = sys.argv[1:]
    stage_cfg = None
    stage_cfg_path = _extract_stage_config_path(raw_argv)

    if stage_cfg_path:
        stage_cfg = _load_stage_config(stage_cfg_path)
        known_fields = _collect_known_parser_fields()
        existing_cli_keys = _collect_cli_keys(raw_argv)
        stage_default_args = _build_stage_default_cli_args(stage_cfg, known_fields, existing_cli_keys)
        if stage_default_args:
            # 前置默认值，保证后面用户传入的同名参数仍然覆盖。
            sys.argv = [sys.argv[0], *stage_default_args, *raw_argv]

    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, MoreConfig))
    sft_script_args, training_args, model_config, more_cfg = parser.parse_args_and_config()

    # 二次兜底：如果 parser 后才拿到 stage_config_path，这里再加载一次。
    if stage_cfg is None and getattr(more_cfg, "stage_config_path", ""):
        stage_cfg = _load_stage_config(more_cfg.stage_config_path)

    if stage_cfg is not None and getattr(more_cfg, "stage_config_path", ""):
        _apply_stage_config(stage_cfg, sft_script_args, training_args, model_config, more_cfg)
        if training_args.local_rank in {0, -1}:
            print(f"[green]Loaded stage config:[/green] {more_cfg.stage_config_path}")
            if getattr(more_cfg, "stage_name", ""):
                print(f"[green]Active stage:[/green] {more_cfg.stage_name}")

    return sft_script_args, training_args, model_config, more_cfg


def _resolve_model_torch_dtype(model_config, training_args):
    """统一计算模型加载 dtype，并校验 flash_attention_2 的硬性约束。"""
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    # FlashAttention2 仅接受 fp16/bf16 输入。
    if model_config.attn_implementation == "flash_attention_2":
        if torch_dtype in ["auto", None]:
            if getattr(training_args, "bf16", False):
                torch_dtype = torch.bfloat16
            elif getattr(training_args, "fp16", False):
                torch_dtype = torch.float16
        if torch_dtype not in [torch.float16, torch.bfloat16]:
            raise ValueError(
                "attn_implementation=flash_attention_2 requires torch_dtype to be fp16 or bf16. "
                "Please set model_arguments.torch_dtype to 'float16' or 'bfloat16'."
            )
    return torch_dtype


def _collect_freeze_patterns(more_cfg, model_name: str):
    """根据配置生成需要冻结的参数名正则。"""
    if "qwen2_vl" not in model_name:
        return []

    patterns = []
    if getattr(more_cfg, "fix_visual_encoder", False):
        patterns.extend([r"visual\.blocks.*", r"visual\.patch_embed.*"])
    if getattr(more_cfg, "fix_visual_adapter", False):
        patterns.append(r"visual\.merger.*")
    if getattr(more_cfg, "fix_language_backbone", False):
        patterns.extend([r"model\.embed_tokens.*", r"model\.layers.*"])
    if getattr(more_cfg, "fix_lm_head", False):
        patterns.extend([r"model\.norm.*", r"lm_head.*"])
    return patterns


def _resolve_image_folder(dataset_name: str, image_folder_override: str):
    """确定 collator 使用的图像根目录。"""
    if image_folder_override:
        return pathlib.Path(image_folder_override)

    dataset_path = pathlib.Path(dataset_name)
    if not dataset_path.exists():
        return pathlib.Path(".")

    image_root = dataset_path.parent
    return image_root.parent if image_root.name == "output" else image_root


def _make_collator_with_dtype_cast(more_cfg, processor, model_name, image_folder, training_args, torch_dtype):
    """构建数据 collator，并在需要时将浮点张量 cast 到模型输入 dtype。"""
    data_collator = make_collator(
        more_cfg.collator_type,
        processor=processor,
        model_path=model_name,
        image_folder=image_folder,
        max_seq_length=training_args.max_seq_length,
        min_pixels=more_cfg.min_pixels,
        max_pixels=more_cfg.max_pixels,
    )

    model_input_dtype = torch_dtype if isinstance(torch_dtype, torch.dtype) else None
    if model_input_dtype not in {torch.float16, torch.bfloat16}:
        return data_collator

    raw_data_collator = data_collator

    def cast_collator(examples):
        batch = raw_data_collator(examples)
        if isinstance(batch, dict):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                    batch[k] = v.to(dtype=model_input_dtype)
        return batch

    if training_args.local_rank in {0, -1}:
        print(f"[green]Batch float tensor cast:[/green] {model_input_dtype}")
    return cast_collator


def _prepare_train_eval_datasets(raw_datasets, more_cfg, training_args):
    """按配置解析 train/eval split，并完成采样比例处理。"""
    train_split = _resolve_dataset_split(
        raw_datasets,
        preferred_name=getattr(more_cfg, "train_split", "train"),
        fallback_candidates=["train"],
    )
    if train_split is None:
        raise ValueError(
            f"Train split not found. Requested '{getattr(more_cfg, 'train_split', 'train')}', "
            f"available splits: {list(raw_datasets.keys())}"
        )

    eval_split = _resolve_dataset_split(
        raw_datasets,
        preferred_name=getattr(more_cfg, "eval_split", "valid"),
        fallback_candidates=["valid", "validation"],
    )

    train_dataset = raw_datasets[train_split]
    train_dataset_len = int(more_cfg.dataset_p * train_dataset.num_rows)
    train_dataset = train_dataset.shuffle(training_args.seed)
    if train_dataset_len < 0:
        select_ids = range(train_dataset.num_rows + train_dataset_len, train_dataset.num_rows)
    else:
        select_ids = range(train_dataset_len)
    train_dataset = train_dataset.select(select_ids)
    eval_dataset = raw_datasets[eval_split] if eval_split is not None else None

    if (training_args.do_eval or training_args.eval_strategy != "no") and eval_dataset is None:
        raise ValueError(
            f"Eval split not found. Requested '{getattr(more_cfg, 'eval_split', 'valid')}', "
            f"available splits: {list(raw_datasets.keys())}"
        )
    return train_dataset, eval_dataset, train_split, eval_split, train_dataset_len


def _apply_namespace_overrides(namespace_obj, overrides: Dict[str, Any], namespace_name: str, strict: bool = False):
    unknown_keys = []
    for key, value in overrides.items():
        if hasattr(namespace_obj, key):
            setattr(namespace_obj, key, value)
        else:
            unknown_keys.append(key)

    if unknown_keys:
        msg = f"Unknown keys in stage config namespace '{namespace_name}': {unknown_keys}"
        if strict:
            raise ValueError(msg)
        print(f"[yellow]{msg}[/yellow]")


def _apply_stage_config(stage_cfg: Dict[str, Any], sft_script_args, training_args, model_config, more_cfg):
    strict = bool(stage_cfg.get("strict", getattr(more_cfg, "strict_stage_config", False)))

    if stage_cfg.get("stage_name") and not getattr(more_cfg, "stage_name", ""):
        more_cfg.stage_name = str(stage_cfg["stage_name"])

    alias_map = {
        "script_arguments": sft_script_args,
        "script_args": sft_script_args,
        "training_arguments": training_args,
        "training_args": training_args,
        "model_arguments": model_config,
        "model_args": model_config,
        "more_arguments": more_cfg,
        "more_args": more_cfg,
    }

    for key, target in alias_map.items():
        overrides = stage_cfg.get(key)
        if isinstance(overrides, dict):
            overrides = dict(overrides)
            # Backward compatibility: some configs place torch_dtype under training_arguments.
            # Model loading dtype is controlled by ModelConfig, so remap it here.
            if key in {"training_arguments", "training_args"} and "torch_dtype" in overrides:
                legacy_dtype = overrides.pop("torch_dtype")
                if not stage_cfg.get("model_arguments", {}).get("torch_dtype") and not stage_cfg.get("model_args", {}).get("torch_dtype"):
                    model_config.torch_dtype = legacy_dtype
                    print("[yellow]Mapped training_arguments.torch_dtype to model_config.torch_dtype for compatibility.[/yellow]")
            _apply_namespace_overrides(target, overrides, key, strict=strict)


def _resolve_dataset_split(raw_datasets, preferred_name: str, fallback_candidates):
    if preferred_name in raw_datasets:
        return preferred_name
    for fallback_name in fallback_candidates:
        if fallback_name in raw_datasets:
            return fallback_name
    return None

if __name__ == "__main__":
    # ===== 第 1 步：参数解析与运行环境准备 =====
    # 统一解析 CLI + stage 配置，保持“显式 CLI 参数优先”。
    sft_script_args, training_args, model_config, more_cfg = _parse_args_with_stage_defaults()

    # 与旧版行为保持一致，避免 gradient checkpointing 的 reentrant 兼容问题。
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    # rich 模式下关闭默认 tqdm，统一日志输出风格。
    if TRL_USE_RICH:
        training_args.disable_tqdm = True

    # 固定随机种子，保证数据采样与训练过程可复现。
    seed_everything(training_args.seed)

    # ===== 第 2 步：构建模型与处理器 =====
    model_name = model_config.model_name_or_path.lower().replace('-', '_')
    torch_dtype = _resolve_model_torch_dtype(model_config, training_args)
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    if training_args.local_rank in {0, -1}:
        print(f"[green]Model load config:[/green] attn_implementation={model_config.attn_implementation}, torch_dtype={torch_dtype}")
    
    # 当前仅支持 Qwen2-VL 家族。
    if 'qwen2_vl' in model_name:
        processor_config = dict(
            do_rescale=False,
            patch_size=14,
            vision_feature_select_strategy="default"
        )
        processor = Qwen2VLProcessor.from_pretrained(model_config.model_name_or_path,**processor_config)
        with open(QWEN_SPECIAL_TOKENS, "r") as file:
            special_token = json.load(file)
        num_new_tokens = processor.tokenizer.add_special_tokens({"additional_special_tokens":special_token})

        model_chat_template = _load_chat_template_from_json(model_config.model_name_or_path)
        if not _is_role_aware_chat_template(processor.tokenizer.chat_template):
            if _is_role_aware_chat_template(model_chat_template):
                processor.tokenizer.chat_template = model_chat_template
            else:
                processor.tokenizer.chat_template = DEFAULT_ROLE_AWARE_QWEN2_VL_CHAT_TEMPLATE

        # 依据 model_kwargs 中配置加载模型（包含 dtype / attention 实现）。
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
        # 只做“扩容”，不做“收缩”：避免把底模预留词表空间裁掉。
        current_vocab_size = model.get_input_embeddings().weight.shape[0]
        target_vocab_size = len(processor.tokenizer)
        if target_vocab_size > current_vocab_size:
            model.resize_token_embeddings(target_vocab_size)
            if training_args.local_rank in {0, -1}:
                print(f"[green]Resized token embeddings:[/green] {current_vocab_size} -> {target_vocab_size} (added={num_new_tokens})")
        elif training_args.local_rank in {0, -1}:
            print(f"[green]Keep token embeddings:[/green] {current_vocab_size} (tokenizer={target_vocab_size}, added={num_new_tokens})")
    else:
        raise ValueError(f"{model_name} unknown")

    # 兜底保证 chat_template 非空，避免 collator 阶段异常。
    if not processor.tokenizer.chat_template:
        processor.tokenizer.chat_template = DEFAULT_ROLE_AWARE_QWEN2_VL_CHAT_TEMPLATE

    # 用一个流式样本验证 chat_template 是否与数据 conversation 结构兼容。
    try:
        sample_split = getattr(more_cfg, "train_split", "train")
        sample_stream = load_dataset(sft_script_args.dataset_name, split=sample_split, streaming=True)
        sample_item = next(iter(sample_stream))
        sample_conversations = sample_item.get("conversations", [])
        rendered = _safe_render_chat_template(processor.tokenizer, sample_conversations)
        if not rendered.strip():
            processor.tokenizer.chat_template = DEFAULT_ROLE_AWARE_QWEN2_VL_CHAT_TEMPLATE
            rendered = _safe_render_chat_template(processor.tokenizer, sample_conversations)
        if not rendered.strip():
            raise ValueError("chat_template is incompatible with dataset conversation schema.")
    except Exception as e:
        raise ValueError(f"Failed to validate chat template compatibility: {e}")
        
    # 右侧 padding 与 pad_token 兜底，保持 batch 对齐行为稳定。
    processor.tokenizer.padding_side = "right"
    if getattr(processor.tokenizer, "pad_token", None) is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # ===== 第 3 步：按配置冻结参数 =====
    # freeze_patterns 是参数名正则列表，命中即冻结。
    freeze_patterns = _collect_freeze_patterns(more_cfg, model_name)

    for name, param in model.named_parameters():
        if any(re.match(pattern, name) for pattern in freeze_patterns):
            param.requires_grad = False

    # ===== 第 4 步：构建 collator 与数据集 =====
    # 若启用 flash/bf16/fp16，会自动将浮点输入对齐到模型 dtype。
    image_fold = _resolve_image_folder(sft_script_args.dataset_name, getattr(more_cfg, "image_folder", ""))
    data_collator = _make_collator_with_dtype_cast(
        more_cfg=more_cfg,
        processor=processor,
        model_name=model_name,
        image_folder=image_fold,
        training_args=training_args,
        torch_dtype=torch_dtype,
    )
    raw_datasets = load_dataset(sft_script_args.dataset_name)

    # 解析 train/eval split，并按 dataset_p 做子集采样。
    train_dataset, eval_dataset, train_split, eval_split, train_dataset_len = _prepare_train_eval_datasets(
        raw_datasets,
        more_cfg,
        training_args,
    )
    
    if training_args.local_rank in { 0 ,-1 }:
        print(train_dataset_len,more_cfg.dataset_p,int(more_cfg.dataset_p*train_dataset_len))
        print(f"[green]Using splits:[/green] train={train_split}, eval={eval_split}")
    
    # ===== 第 5 步：训练器构建与执行 =====
    # 若存在 checkpoint，自动恢复训练。
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        training_args.resume_from_checkpoint = True
        
    # 训练场景关闭 use_cache，避免与 checkpointing 等机制冲突。
    model.config.use_cache = False 

    # 数据已在 collator 中完成组织，告知 Trainer 跳过文本字段预处理。
    training_args.dataset_text_field = "text"
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
        
    trainer = Trainer( 
        model=model,
        args=training_args, 
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.tokenizer,
        model_init=None,
        compute_metrics= None,
        callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        preprocess_logits_for_metrics=None,
    )
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        print_trainable_parameters(trainer.model,trainer.optimizer,record_path=None)

    # 训练/评测分支：优先执行训练；仅当不训练且开启评测时执行 evaluate。
    if training_args.do_train:
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
    elif not training_args.do_train and training_args.do_eval:
        trainer.evaluate()

    # 当 save_strategy 允许保存时，导出最终模型。
    if training_args.save_strategy != "no":
        trainer.save_model(training_args.output_dir)

