'''
Author: Muyao 2350076251@qq.com
Date: 2025-03-04 23:31:28
LastEditors: Muyao 2350076251@qq.com
LastEditTime: 2025-03-05 03:47:11
'''
import random
import torch
import os
import numpy as np
import pathlib
from dataclasses import dataclass, field

IGNORE_TOKEN_ID = -100

@dataclass
class MoreConfig:
    stage_name: str = field(default="", metadata={"help": "Unified stage name: stage1/stage2/stage3"})
    stage_config_path: str = field(default="", metadata={"help": "Path to stage config JSON file"})
    strict_stage_config: bool = field(default=False, metadata={"help": "Raise error on unknown keys in stage config"})
    dataset_p: float = field(default=1.0, metadata={"help": "Dataset parameter p"})
    train_split: str = field(default="train", metadata={"help": "Training split name"})
    eval_split: str = field(default="valid", metadata={"help": "Evaluation split name"})
    image_folder: str = field(default="", metadata={"help": "Optional image folder override for data collator"})
    stage1_train_file: str = field(default="mc-qa-250312.jsonl", metadata={"help": "Stage I train JSONL filename under dataset_name directory"})
    stage1_eval_file: str = field(default="mc-knowledge-valid.jsonl", metadata={"help": "Stage I eval JSONL filename under dataset_name directory"})
    stage1_eval_label_filter: str = field(default="qa", metadata={"help": "Optional label filter for Stage I eval set; empty string disables filtering"})
    collator_type: str = field(default="MultimodalChatDataCollatorforVLM", metadata={"help": "types of collator"})
    fix_visual_encoder: bool = field(default=False, metadata={"help": "fix visual encoder"})
    fix_visual_adapter: bool = field(default=False, metadata={"help": "fix visual adapter layer"})
    fix_language_backbone: bool = field(default=False, metadata={"help": "fix language backbone"})
    fix_lm_head: bool = field(default=False, metadata={"help": "fix language model head"})
    min_pixels: int = field(default=3136)
    max_pixels: int = field(default=2048*28*28)

def seed_everything(seed: int) -> None:
    """Set global random seed for reproducibility."""
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def print_trainable_parameters(model:torch.nn.Module,optimizer:torch.optim.Optimizer=None,record_path = None):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    model_shapes = []

    def get_effective_numel(parameter: torch.nn.Parameter) -> int:
        # Under ZeRO-3, local shard numel can be 0 and the full size is stored in ds_numel.
        local_numel = parameter.numel()
        if local_numel == 0 and hasattr(parameter, "ds_numel"):
            return int(parameter.ds_numel)
        return int(local_numel)

    for name, parameter in model.named_parameters():
        if optimizer:
            optimizer_group_idx = None
            for idx,param_group in enumerate(optimizer.param_groups):
                for param in param_group["params"]:
                    if parameter is param:
                        optimizer_group_idx = idx
            model_shapes.append([parameter.requires_grad,name,parameter.shape,optimizer_group_idx])
        else:
            model_shapes.append([parameter.requires_grad,name,parameter.shape])
        effective_numel = get_effective_numel(parameter)
        all_param += effective_numel
        if parameter.requires_grad:
            trainable_params += effective_numel
    import json
    if record_path:
        pathlib.Path(record_path).parent.mkdir(parents=True,exist_ok=True)
        with open(record_path,mode="w",encoding="UTF-8") as f:
            json.dump(model_shapes, f, indent=4)
        
        with open(record_path.replace(".json","-scratch.txt"),mode="w",encoding="UTF-8") as f:
            print(optimizer, file=f)
            print(model, file=f)
    if all_param == 0:
        print("trainable params: 0 || all params: 0 || trainable%: 0.0")
        return

    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
