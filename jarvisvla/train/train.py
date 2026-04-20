'''
Author: Muyao 2350076251@qq.com
Date: 2025-03-04 23:35:08
LastEditors: Muyao 2350076251@qq.com
LastEditTime: 2025-05-28 23:10:17
'''
import logging
import os
from contextlib import nullcontext
import pathlib

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
    
from datasets import load_dataset,Dataset

import torch
from tqdm.rich import tqdm
from transformers import Qwen2VLProcessor
from transformers import Qwen2VLForConditionalGeneration
from transformers import Trainer

import json
import re
from typing import Any, Dict

from rich import print,console
from jarvisvla.train.utils_train import (
    print_trainable_parameters,
    seed_everything,
    MoreConfig,
)
from jarvisvla import QWEN_SPECIAL_TOKENS
from jarvisvla.train.data_collator import make_collator

tqdm.pandas()    

DEFAULT_QWEN2_VL_CHAT_TEMPLATE = (
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
            _apply_namespace_overrides(target, overrides, key, strict=strict)


def _resolve_dataset_split(raw_datasets, preferred_name: str, fallback_candidates):
    if preferred_name in raw_datasets:
        return preferred_name
    for fallback_name in fallback_candidates:
        if fallback_name in raw_datasets:
            return fallback_name
    return None

if __name__ == "__main__":
    
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, MoreConfig))
    sft_script_args, training_args, model_config, more_cfg = parser.parse_args_and_config()

    if getattr(more_cfg, "stage_config_path", ""):
        stage_cfg = _load_stage_config(more_cfg.stage_config_path)
        _apply_stage_config(stage_cfg, sft_script_args, training_args, model_config, more_cfg)
        if training_args.local_rank in {0, -1}:
            print(f"[green]Loaded stage config:[/green] {more_cfg.stage_config_path}")
            if getattr(more_cfg, "stage_name", ""):
                print(f"[green]Active stage:[/green] {more_cfg.stage_name}")

    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    seed_everything(training_args.seed)

    ################
    # Model, Tokenizer & Processor
    ################
    
    model_name = model_config.model_name_or_path.lower().replace('-','_')
    
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    
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
                processor.tokenizer.chat_template = DEFAULT_QWEN2_VL_CHAT_TEMPLATE

        model_kwargs["attn_implementation"] = "flash_attention_2"
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
        if num_new_tokens > 0:
            model.resize_token_embeddings(len(processor.tokenizer))
    else:
        raise ValueError(f"{model_name} unknown")

    if not processor.tokenizer.chat_template:
        processor.tokenizer.chat_template = DEFAULT_QWEN2_VL_CHAT_TEMPLATE

    # Verify the chat template can render at least one training sample.
    try:
        sample_split = getattr(more_cfg, "train_split", "train")
        sample_stream = load_dataset(sft_script_args.dataset_name, split=sample_split, streaming=True)
        sample_item = next(iter(sample_stream))
        sample_conversations = sample_item.get("conversations", [])
        rendered = _safe_render_chat_template(processor.tokenizer, sample_conversations)
        if not rendered.strip():
            processor.tokenizer.chat_template = DEFAULT_QWEN2_VL_CHAT_TEMPLATE
            rendered = _safe_render_chat_template(processor.tokenizer, sample_conversations)
        if not rendered.strip():
            raise ValueError("chat_template is incompatible with dataset conversation schema.")
    except Exception as e:
        raise ValueError(f"Failed to validate chat template compatibility: {e}")
        
    processor.tokenizer.padding_side = "right"
    if getattr(processor.tokenizer, "pad_token", None) is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    fix_refexs = []
    if getattr(more_cfg, 'fix_visual_encoder', False):
        if 'qwen2_vl' in model_name:
            fix_refexs.append(r"visual\.blocks.*")
            fix_refexs.append(r"visual\.patch_embed.*")
    if getattr(more_cfg, 'fix_visual_adapter', False):
        if 'qwen2_vl' in model_name:
            fix_refexs.append(r"visual\.merger.*")
    if getattr(more_cfg, 'fix_language_backbone', False):
        if 'qwen2_vl' in model_name:
            fix_refexs.append(r"model\.embed_tokens.*")
            fix_refexs.append(r"model\.layers.*")
    if getattr(more_cfg, 'fix_lm_head', False):
        if 'qwen2_vl' in model_name:
            fix_refexs.append(r"model\.norm.*")
            fix_refexs.append(r"lm_head.*")
       
    for name, param in model.named_parameters():
        if any(re.match(pattern, name) for pattern in fix_refexs):
            param.requires_grad = False
    
    
    ##################
    # DataCollator
    ##################

    # Resolve image folder for local datasets; allow explicit override from stage config.
    if getattr(more_cfg, "image_folder", ""):
        image_fold = pathlib.Path(more_cfg.image_folder)
    else:
        dataset_path = pathlib.Path(sft_script_args.dataset_name)
        if dataset_path.exists():
            image_fold = dataset_path.parent
            image_fold = image_fold.parent if image_fold.name == "output" else image_fold
        else:
            image_fold = pathlib.Path(".")

    data_collator = make_collator(more_cfg.collator_type, 
                                  processor=processor, 
                                  model_path=model_name,
                                  image_folder=image_fold,
                                  max_seq_length = training_args.max_seq_length,
                                  min_pixels = more_cfg.min_pixels,
                                  max_pixels = more_cfg.max_pixels,
                                  )
    
    ################
    # Dataset
    ################
    
    raw_datasets = load_dataset(sft_script_args.dataset_name)

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
    train_dataset_len = train_dataset.num_rows
    train_dataset_len = int(more_cfg.dataset_p*train_dataset_len)
    train_dataset = train_dataset.shuffle(training_args.seed)
    if train_dataset_len < 0:
        select_ids = range(train_dataset.num_rows + train_dataset_len,train_dataset.num_rows)
    else:
        select_ids = range(train_dataset_len)
    train_dataset = train_dataset.select(select_ids) 
    eval_dataset = raw_datasets[eval_split] if eval_split is not None else None

    if (training_args.do_eval or training_args.eval_strategy != "no") and eval_dataset is None:
        raise ValueError(
            f"Eval split not found. Requested '{getattr(more_cfg, 'eval_split', 'valid')}', "
            f"available splits: {list(raw_datasets.keys())}"
        )
    
    if training_args.local_rank in { 0 ,-1 }:
        print(train_dataset_len,more_cfg.dataset_p,int(more_cfg.dataset_p*train_dataset_len))
        print(f"[green]Using splits:[/green] train={train_split}, eval={eval_split}")
    
    ################
    # Optional rich context managers
    ###############
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Training
    ################
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        training_args.resume_from_checkpoint = True
        
    # Ensure use_cache is set to False
    model.config.use_cache = False 

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

    if training_args.do_train:
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
    elif not training_args.do_train and training_args.do_eval:
        trainer.evaluate()

    if training_args.save_strategy != "no":
        trainer.save_model(training_args.output_dir)

