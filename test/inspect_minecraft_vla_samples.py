#!/usr/bin/env python3
"""Inspect concrete sample contents in minecraft-vla-sft.

This script focuses on qualitative inspection of `label` and `conversations`
to help verify whether dataset contents are informative.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect sample contents from minecraft-vla-sft.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="/share/public_datasets/VLA/nitrogen/minecraft-vla-sft",
        help="Dataset path accepted by datasets.load_dataset().",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Which split to inspect: train/valid.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="How many samples to inspect.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index when not using random sampling.",
    )
    parser.add_argument(
        "--random-sample",
        action="store_true",
        help="Randomly sample indices instead of contiguous slice.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=43,
        help="Random seed used when --random-sample is enabled.",
    )
    parser.add_argument(
        "--text-max-chars",
        type=int,
        default=180,
        help="Max chars to print per text snippet.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output json path for the inspected samples and summary.",
    )
    return parser.parse_args()


def shorten(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def choose_indices(total: int, num_samples: int, start_index: int, random_sample: bool, seed: int) -> list[int]:
    if total <= 0 or num_samples <= 0:
        return []
    k = min(num_samples, total)
    if random_sample:
        rng = random.Random(seed)
        return sorted(rng.sample(range(total), k))

    start = max(0, min(start_index, total - 1))
    end = min(start + k, total)
    return list(range(start, end))


def normalize_content_items(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, list):
        out: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, dict):
                out.append(item)
            else:
                out.append({"type": "unknown", "text": to_text(item)})
        return out
    if isinstance(content, dict):
        return [content]
    return [{"type": "text", "text": to_text(content)}]


def inspect_sample(sample: dict[str, Any], sample_index: int, text_max_chars: int) -> dict[str, Any]:
    labels = sample.get("label")
    labels = labels if isinstance(labels, list) else ([] if labels is None else [to_text(labels)])

    conversations = sample.get("conversations")
    conversations = conversations if isinstance(conversations, list) else []

    turn_summaries: list[dict[str, Any]] = []
    role_sequence: list[str] = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = to_text(turn.get("role", "unknown")).strip().lower() or "unknown"
        role_sequence.append(role)
        items = normalize_content_items(turn.get("content"))
        content_summaries = []
        for item in items:
            item_type = to_text(item.get("type", "unknown")).strip() or "unknown"
            item_text = shorten(to_text(item.get("text", "")), text_max_chars)
            content_summaries.append({"type": item_type, "text": item_text})
        turn_summaries.append(
            {
                "role": role,
                "num_content_items": len(content_summaries),
                "content": content_summaries,
            }
        )

    image = sample.get("image")
    image_count = len(image) if isinstance(image, list) else (0 if image is None else 1)

    image_bytes = sample.get("image_bytes")
    image_bytes_count = len(image_bytes) if isinstance(image_bytes, list) else (0 if image_bytes is None else 1)

    return {
        "sample_index": sample_index,
        "id": sample.get("id"),
        "label": labels,
        "num_label_items": len(labels),
        "image_count": image_count,
        "image_bytes_count": image_bytes_count,
        "num_turns": len(turn_summaries),
        "role_sequence": role_sequence,
        "conversations": turn_summaries,
    }


def main() -> None:
    args = parse_args()

    print(f"[INFO] Loading split '{args.split}' from: {args.dataset}")
    dataset = load_dataset(args.dataset, split=args.split)
    total = int(dataset.num_rows)
    print(f"[INFO] Total rows in split '{args.split}': {total}")

    indices = choose_indices(
        total=total,
        num_samples=args.num_samples,
        start_index=args.start_index,
        random_sample=args.random_sample,
        seed=args.seed,
    )
    if not indices:
        print("[WARN] No indices selected. Check --num-samples and dataset size.")
        return

    selected = dataset.select(indices)

    role_counter: Counter[str] = Counter()
    content_type_counter: Counter[str] = Counter()
    label_item_counter: Counter[str] = Counter()
    num_turns_counter: Counter[int] = Counter()

    inspected_samples: list[dict[str, Any]] = []

    for row_idx, sample in zip(indices, selected):
        inspected = inspect_sample(sample, row_idx, args.text_max_chars)
        inspected_samples.append(inspected)

        for label_item in inspected["label"]:
            label_item_counter[to_text(label_item)] += 1

        num_turns_counter[inspected["num_turns"]] += 1
        for turn in inspected["conversations"]:
            role_counter[turn["role"]] += 1
            for content in turn["content"]:
                content_type_counter[content["type"]] += 1

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "total_rows": total,
        "selected_indices": indices,
        "sampled_size": len(inspected_samples),
        "num_turns_distribution": dict(num_turns_counter),
        "role_distribution": dict(role_counter),
        "content_type_distribution": dict(content_type_counter),
        "top_label_items": label_item_counter.most_common(30),
    }

    print("\n[SUMMARY]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n[SAMPLE DETAILS]")
    for item in inspected_samples:
        print("-" * 80)
        print(f"index={item['sample_index']} id={item['id']}")
        print(f"label({item['num_label_items']}): {item['label']}")
        print(f"turns={item['num_turns']} roles={item['role_sequence']}")
        for t_i, turn in enumerate(item["conversations"]):
            print(f"  turn[{t_i}] role={turn['role']} content_items={turn['num_content_items']}")
            for c_i, content in enumerate(turn["content"]):
                print(f"    content[{c_i}] type={content['type']} text={content['text']}")

    if args.output:
        payload = {"summary": summary, "samples": inspected_samples}
        output_path = Path(args.output)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[INFO] Wrote inspection report to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
