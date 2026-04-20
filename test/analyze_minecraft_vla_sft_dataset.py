#!/usr/bin/env python3
"""Analyze the 3rd label (index=2) with major/minor hierarchy.

This script computes, for each split:
1) number of processed samples
2) count of the 3rd label major category (before ':')
3) count of minor category (after ':') within each major category
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze label-by-position statistics.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="/share/public_datasets/VLA/nitrogen/minecraft-vla-sft",
        help="Dataset path accepted by datasets.load_dataset().",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=0,
        help="If > 0, only process this many samples per split.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N samples. Set <=0 to disable periodic progress.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size used in PyTorch DataLoader mode.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count. Set >0 to enable PyTorch multi-process loading.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output text path. Default: ./logs/label_hierarchy_report_<timestamp>.txt",
    )
    parser.add_argument(
        "--analyze-conversations",
        action="store_true",
        help="Extract plain-text user instructions from conversations and aggregate by subcategory."
        " When enabled, output will be JSON.",
    )
    return parser.parse_args()


def to_text(value: Any) -> str:
    if value is None:
        return "<NULL>"
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def format_seconds(seconds: float) -> str:
    if seconds < 0:
        return "unknown"
    seconds = int(seconds)
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def collate_samples(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item if isinstance(item, dict) else {} for item in batch]


def normalize_image_source_dir(image_path: str) -> str:
    # Normalize sharded image folders like .../images/s/file.jpg to .../images.
    p = Path(image_path)
    parent = p.parent
    leaf = parent.name
    if len(leaf) == 1 and leaf.isalnum():
        parent = parent.parent
    return str(parent)


def extract_image_source_dirs(image_field: Any) -> list[str]:
    out: list[str] = []
    if isinstance(image_field, str):
        if image_field:
            out.append(image_field)
    elif isinstance(image_field, list):
        for item in image_field:
            if isinstance(item, str) and item:
                out.append(item)
    return out


def extract_plain_instruction(conversations: Any) -> str:
    if not isinstance(conversations, list):
        return ""

    user_text_parts: list[str] = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = to_text(turn.get("role")).strip().lower()
        if role != "user":
            continue

        content = turn.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = to_text(item.get("type")).strip().lower()
                if item_type == "image":
                    continue
                item_text = to_text(item.get("text")).strip()
                if item_text:
                    user_text_parts.append(item_text)
        else:
            text = to_text(content).strip()
            if text:
                user_text_parts.append(text)

    if not user_text_parts:
        return ""

    raw = "\n".join(user_text_parts)
    raw = raw.replace("<image>", " ")
    raw = re.sub(r"<\|[^>]+\|>", " ", raw)
    raw = re.split(r"\bobservation\s*:", raw, maxsplit=1, flags=re.IGNORECASE)[0]
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def update_counters_from_label(
    labels: Any,
    major_counter: Counter[str],
    minor_counter_by_major: dict[str, Counter[str]],
) -> tuple[bool, bool, str | None, str | None]:
    if not isinstance(labels, list):
        return True, False, None, None

    if len(labels) <= 2:
        return False, True, None, None

    raw_third = to_text(labels[2]).strip()
    if ":" in raw_third:
        major, minor = raw_third.split(":", 1)
        major = major.strip() or "<EMPTY_MAJOR>"
        minor = minor.strip() or "<EMPTY_MINOR>"
        if major not in minor_counter_by_major:
            minor_counter_by_major[major] = Counter()
        major_counter[major] += 1
        minor_counter_by_major[major][minor] += 1
        return False, False, None, f"{major}:{minor}"
    else:
        malformed_raw = raw_third if raw_third else "<EMPTY_RAW_THIRD_LABEL>"
        return False, False, malformed_raw, None


def analyze_split(
    dataset: Any,
    split_name: str,
    max_samples: int,
    progress_every: int,
    batch_size: int,
    num_workers: int,
    analyze_conversations: bool,
) -> dict[str, Any]:
    total_rows = int(dataset.num_rows)
    limit = min(max_samples, total_rows) if max_samples and max_samples > 0 else total_rows

    major_counter: Counter[str] = Counter()
    minor_counter_by_major: dict[str, Counter[str]] = {}
    missing_label_rows = 0
    missing_third_label_rows = 0
    malformed_third_label_rows = 0
    malformed_third_label_details: Counter[str] = Counter()
    subcategory_sample_counter: Counter[str] = Counter()
    instruction_counter_by_subcategory: dict[str, Counter[str]] = {}
    malformed_sample_counter: Counter[str] = Counter()
    instruction_counter_by_malformed_label: dict[str, Counter[str]] = {}
    empty_instruction_rows = 0
    malformed_empty_instruction_rows = 0
    image_path_count = 0
    samples_with_image = 0
    samples_without_image = 0
    image_dir_raw_counter: Counter[str] = Counter()
    image_dir_normalized_counter: Counter[str] = Counter()
    analyzed_rows = 0

    start_ts = time.time()
    last_print_ts = start_ts

    def print_progress(force: bool = False) -> None:
        nonlocal last_print_ts
        if analyzed_rows == 0:
            return
        now = time.time()
        should_print = force
        if not should_print and progress_every > 0:
            should_print = (analyzed_rows % progress_every == 0)
        if not should_print:
            return
        if not force and now - last_print_ts < 0.8:
            return

        elapsed = max(now - start_ts, 1e-9)
        speed = analyzed_rows / elapsed
        remaining = max(limit - analyzed_rows, 0)
        eta = (remaining / speed) if speed > 0 else -1.0
        pct = 100.0 * analyzed_rows / limit if limit > 0 else 100.0
        print(
            "[PROGRESS] split='{}' {}/{} ({:.2f}%) | {:.1f} samples/s | elapsed {} | ETA {}".format(
                split_name,
                analyzed_rows,
                limit,
                pct,
                speed,
                format_seconds(elapsed),
                format_seconds(eta),
            )
        )
        last_print_ts = now

    if num_workers > 0:
        try:
            from torch.utils.data import DataLoader

            print(
                f"[INFO] split='{split_name}': using PyTorch DataLoader parallel mode "
                f"(num_workers={num_workers}, batch_size={batch_size})"
            )
            dataset_slice = dataset if limit == total_rows else dataset.select(range(limit))
            keep_cols = {"label", "image"}
            if analyze_conversations:
                keep_cols.add("conversations")
            drop_cols = [c for c in dataset_slice.column_names if c not in keep_cols]
            if drop_cols:
                dataset_slice = dataset_slice.remove_columns(drop_cols)

            loader = DataLoader(
                dataset_slice,
                batch_size=max(1, batch_size),
                num_workers=num_workers,
                shuffle=False,
                collate_fn=collate_samples,
                persistent_workers=num_workers > 0,
            )

            for sample_batch in loader:
                for sample in sample_batch:
                    analyzed_rows += 1

                    image_paths = extract_image_source_dirs(sample.get("image"))
                    if image_paths:
                        samples_with_image += 1
                    else:
                        samples_without_image += 1
                    for img_path in image_paths:
                        image_path_count += 1
                        raw_dir = str(Path(img_path).parent)
                        image_dir_raw_counter[raw_dir] += 1
                        image_dir_normalized_counter[normalize_image_source_dir(img_path)] += 1

                    miss_label, miss_third, malformed_raw, subcategory_key = update_counters_from_label(
                        sample.get("label"),
                        major_counter,
                        minor_counter_by_major,
                    )
                    if miss_label:
                        missing_label_rows += 1
                    elif miss_third:
                        missing_third_label_rows += 1
                    elif malformed_raw is not None:
                        malformed_third_label_rows += 1
                        malformed_third_label_details[malformed_raw] += 1
                        if analyze_conversations:
                            malformed_sample_counter[malformed_raw] += 1
                            if malformed_raw not in instruction_counter_by_malformed_label:
                                instruction_counter_by_malformed_label[malformed_raw] = Counter()
                            instruction = extract_plain_instruction(sample.get("conversations"))
                            if instruction:
                                instruction_counter_by_malformed_label[malformed_raw][instruction] += 1
                            else:
                                malformed_empty_instruction_rows += 1

                    if analyze_conversations and subcategory_key is not None:
                        subcategory_sample_counter[subcategory_key] += 1
                        if subcategory_key not in instruction_counter_by_subcategory:
                            instruction_counter_by_subcategory[subcategory_key] = Counter()
                        instruction = extract_plain_instruction(sample.get("conversations"))
                        if instruction:
                            instruction_counter_by_subcategory[subcategory_key][instruction] += 1
                        else:
                            empty_instruction_rows += 1

                print_progress(force=False)

        except Exception as e:
            print(
                f"[WARN] split='{split_name}': PyTorch parallel mode unavailable ({e}). "
                "Falling back to sequential mode."
            )
            num_workers = 0

    if num_workers <= 0:
        print(f"[INFO] split='{split_name}': using sequential mode")
        for i, sample in enumerate(dataset):
            if i >= limit:
                break
            analyzed_rows += 1

            image_paths = extract_image_source_dirs(sample.get("image") if isinstance(sample, dict) else None)
            if image_paths:
                samples_with_image += 1
            else:
                samples_without_image += 1
            for img_path in image_paths:
                image_path_count += 1
                raw_dir = str(Path(img_path).parent)
                image_dir_raw_counter[raw_dir] += 1
                image_dir_normalized_counter[normalize_image_source_dir(img_path)] += 1

            miss_label, miss_third, malformed_raw, subcategory_key = update_counters_from_label(
                sample.get("label") if isinstance(sample, dict) else None,
                major_counter,
                minor_counter_by_major,
            )
            if miss_label:
                missing_label_rows += 1
            elif miss_third:
                missing_third_label_rows += 1
            elif malformed_raw is not None:
                malformed_third_label_rows += 1
                malformed_third_label_details[malformed_raw] += 1
                if analyze_conversations:
                    malformed_sample_counter[malformed_raw] += 1
                    if malformed_raw not in instruction_counter_by_malformed_label:
                        instruction_counter_by_malformed_label[malformed_raw] = Counter()
                    instruction = extract_plain_instruction(sample.get("conversations") if isinstance(sample, dict) else None)
                    if instruction:
                        instruction_counter_by_malformed_label[malformed_raw][instruction] += 1
                    else:
                        malformed_empty_instruction_rows += 1

            if analyze_conversations and subcategory_key is not None:
                subcategory_sample_counter[subcategory_key] += 1
                if subcategory_key not in instruction_counter_by_subcategory:
                    instruction_counter_by_subcategory[subcategory_key] = Counter()
                instruction = extract_plain_instruction(sample.get("conversations") if isinstance(sample, dict) else None)
                if instruction:
                    instruction_counter_by_subcategory[subcategory_key][instruction] += 1
                else:
                    empty_instruction_rows += 1

            print_progress(force=False)

    print_progress(force=True)

    major_sorted = [[k, int(v)] for k, v in sorted(major_counter.items(), key=lambda kv: (-kv[1], kv[0]))]
    minor_sorted_by_major: dict[str, list[list[Any]]] = {}
    for major, counter in minor_counter_by_major.items():
        minor_sorted_by_major[major] = [
            [k, int(v)] for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
    malformed_details_sorted = [
        [k, int(v)] for k, v in sorted(malformed_third_label_details.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    image_dir_raw_sorted = [
        [k, int(v)] for k, v in sorted(image_dir_raw_counter.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    image_dir_normalized_sorted = [
        [k, int(v)] for k, v in sorted(image_dir_normalized_counter.items(), key=lambda kv: (-kv[1], kv[0]))
    ]

    instruction_stats_by_subcategory: dict[str, Any] = {}
    if analyze_conversations:
        for key in sorted(subcategory_sample_counter):
            counter = instruction_counter_by_subcategory.get(key, Counter())
            instruction_stats_by_subcategory[key] = {
                "samples": int(subcategory_sample_counter[key]),
                "unique_instructions": len(counter),
                "empty_instruction_rows": int(subcategory_sample_counter[key] - sum(counter.values())),
                "instruction_counts": [
                    [text, int(cnt)] for text, cnt in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
                ],
            }

    malformed_instruction_stats: dict[str, Any] = {}
    if analyze_conversations:
        for raw_label in sorted(malformed_sample_counter):
            counter = instruction_counter_by_malformed_label.get(raw_label, Counter())
            malformed_instruction_stats[raw_label] = {
                "samples": int(malformed_sample_counter[raw_label]),
                "unique_instructions": len(counter),
                "empty_instruction_rows": int(malformed_sample_counter[raw_label] - sum(counter.values())),
                "instruction_counts": [
                    [text, int(cnt)] for text, cnt in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
                ],
            }

    result = {
        "total_rows": total_rows,
        "analyzed_rows": analyzed_rows,
        "missing_label_rows": missing_label_rows,
        "missing_third_label_rows": missing_third_label_rows,
        "malformed_third_label_rows": malformed_third_label_rows,
        "malformed_third_label_unique_values": len(malformed_third_label_details),
        "malformed_third_label_details": malformed_details_sorted,
        "image_source_stats": {
            "samples_with_image": samples_with_image,
            "samples_without_image": samples_without_image,
            "image_path_count": image_path_count,
            "raw_image_dir_counts": image_dir_raw_sorted,
            "normalized_image_source_dir_counts": image_dir_normalized_sorted,
        },
        "major_category_counts": major_sorted,
        "minor_category_counts_by_major": minor_sorted_by_major,
    }
    if analyze_conversations:
        result["conversation_instruction_summary"] = {
            "enabled": True,
            "empty_instruction_rows": empty_instruction_rows,
            "subcategory_count": len(subcategory_sample_counter),
        }
        result["instruction_stats_by_subcategory"] = instruction_stats_by_subcategory
        result["malformed_conversation_instruction_summary"] = {
            "enabled": True,
            "malformed_label_count": len(malformed_sample_counter),
            "empty_instruction_rows": malformed_empty_instruction_rows,
        }
        result["instruction_stats_for_malformed_labels"] = malformed_instruction_stats
    return result


def main() -> None:
    args = parse_args()

    print(f"[INFO] Loading dataset from: {args.dataset}")
    raw = load_dataset(args.dataset)
    split_names = list(raw.keys())
    print(f"[INFO] Splits found: {split_names}")

    report: dict[str, Any] = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "dataset_path": args.dataset,
        "split_names": split_names,
        "analysis": (
            "processed_rows_and_3rd_label_major_minor_hierarchy_with_conversation_instructions"
            if args.analyze_conversations
            else "processed_rows_and_3rd_label_major_minor_hierarchy"
        ),
        "splits": {},
    }

    for split_name in split_names:
        print(f"[INFO] Analyzing split='{split_name}' ...")
        split_report = analyze_split(
            dataset=raw[split_name],
            split_name=split_name,
            max_samples=args.max_samples_per_split,
            progress_every=args.progress_every,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            analyze_conversations=args.analyze_conversations,
        )
        report["splits"][split_name] = split_report
        print(
            "[INFO] Done split='{}': total_rows={}, analyzed_rows={}, missing_label_rows={}, missing_third_label_rows={}, malformed_third_label_rows={}".format(
                split_name,
                split_report["total_rows"],
                split_report["analyzed_rows"],
                split_report["missing_label_rows"],
                split_report["missing_third_label_rows"],
                split_report["malformed_third_label_rows"],
            )
        )

    if args.analyze_conversations:
        output_path = Path(args.output) if args.output else Path(
            f"logs/label_hierarchy_with_instructions_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    else:
        output_path = Path(args.output) if args.output else Path(
            f"logs/label_hierarchy_report_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )

    if args.analyze_conversations:
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] Report written to: {output_path.resolve()}")
        return

    lines: list[str] = []
    lines.append(f"generated_at: {report['generated_at']}")
    lines.append(f"dataset_path: {report['dataset_path']}")
    lines.append(f"analysis: {report['analysis']}")
    lines.append(f"splits: {', '.join(report['split_names'])}")
    lines.append("")

    for split_name in report["split_names"]:
        s = report["splits"][split_name]
        lines.append(f"=== split: {split_name} ===")
        lines.append(f"total_rows: {s['total_rows']}")
        lines.append(f"analyzed_rows: {s['analyzed_rows']}")
        lines.append(f"missing_label_rows: {s['missing_label_rows']}")
        lines.append(f"missing_third_label_rows: {s['missing_third_label_rows']}")
        lines.append(f"malformed_third_label_rows: {s['malformed_third_label_rows']}")
        lines.append(f"malformed_third_label_unique_values: {s['malformed_third_label_unique_values']}")
        lines.append("")
        lines.append("[image_source_stats]")
        lines.append(f"samples_with_image: {s['image_source_stats']['samples_with_image']}")
        lines.append(f"samples_without_image: {s['image_source_stats']['samples_without_image']}")
        lines.append(f"image_path_count: {s['image_source_stats']['image_path_count']}")
        lines.append("")
        lines.append("[raw_image_dir_counts]")
        for src_dir, cnt in s["image_source_stats"]["raw_image_dir_counts"]:
            lines.append(f"- {src_dir}: {cnt}")
        lines.append("")
        lines.append("[major_category_counts]")
        for major, cnt in s["major_category_counts"]:
            lines.append(f"- {major}: {cnt}")
        lines.append("")
        lines.append("[minor_category_counts_by_major]")
        for major, minor_list in s["minor_category_counts_by_major"].items():
            lines.append(f"{major}:")
            for minor, cnt in minor_list:
                lines.append(f"  - {minor}: {cnt}")
        lines.append("")
        lines.append("[malformed_third_label_details]")
        if s["malformed_third_label_details"]:
            for raw_value, cnt in s["malformed_third_label_details"]:
                lines.append(f"- {raw_value}: {cnt}")
        else:
            lines.append("- <NONE>: 0")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[INFO] Report written to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
