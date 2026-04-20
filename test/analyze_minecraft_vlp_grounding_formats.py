#!/usr/bin/env python3
"""Analyze whether minecraft-vlp grounding samples use point or bbox supervision.

This script inspects grounding-style JSONL files in minecraft-vlp and answers three
questions:

1. Do any assistant answers use structured `bbox` outputs?
2. Do the prompts explicitly ask for `bbox` or `point` outputs?
3. Are there samples whose prompt suggests bbox but whose answer is point, or vice versa?

The goal is to distinguish between:

- direct supervision format used in the sample answers
- auxiliary metadata such as `source.bbox` / `source.points`
- question wording that may imply one output format or the other
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DATASET_ROOT = "/share/public_datasets/VLA/nitrogen/minecraft-vlp"
DEFAULT_FILES = [
    "mc-grounding-point-gui.jsonl",
    "mc-grounding-point-embodied.jsonl",
    "mc-grounding-point-embodied-image5.jsonl",
    "mc-knowledge-valid.jsonl",
    "hallucination.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze grounding supervision formats in minecraft-vlp.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory of minecraft-vlp.",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Specific JSONL files to analyze. Can be passed multiple times. Defaults to grounding-related files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output JSON path for the analysis report.",
    )
    parser.add_argument(
        "--max-samples-per-file",
        type=int,
        default=0,
        help="If > 0, only inspect the first N rows per file.",
    )
    parser.add_argument(
        "--example-limit",
        type=int,
        default=5,
        help="How many representative examples to retain per category.",
    )
    return parser.parse_args()


def to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def normalize_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_first_user_prompt(conversations: Any) -> str:
    if not isinstance(conversations, list):
        return ""
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        if to_text(turn.get("role")).strip().lower() != "user":
            continue
        content = turn.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = to_text(item.get("type")).strip().lower()
                if item_type == "image":
                    continue
                item_text = normalize_text(to_text(item.get("text")))
                if item_text:
                    parts.append(item_text)
            if parts:
                return normalize_text(" ".join(parts))
        else:
            prompt = normalize_text(to_text(content))
            if prompt:
                return prompt
    return ""


def iter_assistant_items(conversations: Any):
    if not isinstance(conversations, list):
        return
    for turn_index, turn in enumerate(conversations):
        if not isinstance(turn, dict):
            continue
        if to_text(turn.get("role")).strip().lower() != "assistant":
            continue
        content = turn.get("content")
        if isinstance(content, list):
            for item_index, item in enumerate(content):
                if isinstance(item, dict):
                    yield turn_index, item_index, item


def classify_prompt(prompt: str) -> str:
    low = prompt.lower()
    asks_bbox = bool(re.search(r"\bbbox\b|bounding box", low))
    asks_point = bool(re.search(r"\bpoint\b|\bpoints\b|point format", low))
    if asks_bbox and asks_point:
        return "bbox_and_point"
    if asks_bbox:
        return "bbox"
    if asks_point:
        return "point"
    return "other"


def summarize_answer_item(item: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "type": to_text(item.get("type")).strip() or "unknown",
    }
    if "text" in item:
        summary["text"] = normalize_text(to_text(item.get("text")))
    if "point" in item:
        summary["point"] = item.get("point")
    if "bbox" in item:
        summary["bbox"] = item.get("bbox")
    if "label" in item:
        summary["label"] = item.get("label")
    return summary


def summarize_source(source: Any) -> dict[str, Any]:
    if not isinstance(source, dict):
        return {}
    summary: dict[str, Any] = {
        "keys": sorted(source.keys()),
    }
    if "points" in source:
        summary["points"] = source.get("points")
    if "bbox" in source:
        summary["bbox"] = source.get("bbox")
    if "label" in source:
        summary["label"] = source.get("label")
    if "image_url" in source:
        summary["image_url"] = source.get("image_url")
    if "image_urls" in source:
        summary["image_urls"] = source.get("image_urls")
    if "video_path" in source:
        summary["video_path"] = source.get("video_path")
    return summary


def load_records(jsonl_path: Path, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if limit > 0 and row_index >= limit:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
    return records


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    file_names = args.file if args.file else list(DEFAULT_FILES)
    file_paths = [dataset_root / name for name in file_names]

    report: dict[str, Any] = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root),
        "files": {},
    }

    global_answer_type_counts = Counter()
    global_prompt_class_counts = Counter()
    global_prompt_vs_answer = defaultdict(Counter)
    global_source_key_counts = Counter()

    for file_path in file_paths:
        if not file_path.exists():
            print(f"[WARN] missing file: {file_path}")
            continue

        records = load_records(file_path, args.max_samples_per_file)
        file_report = {
            "records": len(records),
            "answer_type_counts": Counter(),
            "prompt_class_counts": Counter(),
            "prompt_vs_answer": defaultdict(Counter),
            "source_key_counts": Counter(),
            "examples": {
                "bbox_answer": [],
                "point_answer": [],
                "prompt_bbox_answer_point": [],
                "prompt_point_answer_bbox": [],
            },
        }

        for sample_index, sample in enumerate(records):
            prompt = get_first_user_prompt(sample.get("conversations"))
            prompt_class = classify_prompt(prompt)
            file_report["prompt_class_counts"][prompt_class] += 1
            global_prompt_class_counts[prompt_class] += 1

            source = sample.get("source")
            if isinstance(source, dict):
                file_report["source_key_counts"].update(source.keys())
                global_source_key_counts.update(source.keys())

            assistant_items = list(iter_assistant_items(sample.get("conversations")))
            answer_types = []
            for turn_index, item_index, item in assistant_items:
                item_type = to_text(item.get("type")).strip().lower() or "unknown"
                answer_types.append(item_type)
                file_report["answer_type_counts"][item_type] += 1
                global_answer_type_counts[item_type] += 1
                file_report["prompt_vs_answer"][prompt_class][item_type] += 1
                global_prompt_vs_answer[prompt_class][item_type] += 1

            has_bbox = any(t == "bbox" for t in answer_types)
            has_point = any(t == "point" for t in answer_types)

            sample_summary = {
                "sample_index": sample_index,
                "id": sample.get("id"),
                "label": sample.get("label"),
                "prompt": prompt,
                "prompt_class": prompt_class,
                "assistant_answer_types": sorted(set(answer_types)),
                "source": summarize_source(sample.get("source")),
            }

            if has_bbox and len(file_report["examples"]["bbox_answer"]) < args.example_limit:
                file_report["examples"]["bbox_answer"].append(sample_summary)
            if has_point and len(file_report["examples"]["point_answer"]) < args.example_limit:
                file_report["examples"]["point_answer"].append(sample_summary)
            if prompt_class == "bbox" and has_point and len(file_report["examples"]["prompt_bbox_answer_point"]) < args.example_limit:
                file_report["examples"]["prompt_bbox_answer_point"].append(sample_summary)
            if prompt_class == "point" and has_bbox and len(file_report["examples"]["prompt_point_answer_bbox"]) < args.example_limit:
                file_report["examples"]["prompt_point_answer_bbox"].append(sample_summary)

        report["files"][file_path.name] = {
            "records": file_report["records"],
            "answer_type_counts": [[k, int(v)] for k, v in file_report["answer_type_counts"].most_common()],
            "prompt_class_counts": [[k, int(v)] for k, v in file_report["prompt_class_counts"].most_common()],
            "prompt_vs_answer": {
                prompt_class: [[k, int(v)] for k, v in counter.most_common()]
                for prompt_class, counter in file_report["prompt_vs_answer"].items()
            },
            "source_key_counts": [[k, int(v)] for k, v in file_report["source_key_counts"].most_common()],
            "examples": file_report["examples"],
        }

    report["global_summary"] = {
        "answer_type_counts": [[k, int(v)] for k, v in global_answer_type_counts.most_common()],
        "prompt_class_counts": [[k, int(v)] for k, v in global_prompt_class_counts.most_common()],
        "prompt_vs_answer": {
            prompt_class: [[k, int(v)] for k, v in counter.most_common()]
            for prompt_class, counter in global_prompt_vs_answer.items()
        },
        "source_key_counts": [[k, int(v)] for k, v in global_source_key_counts.most_common()],
    }

    output_path = Path(args.output) if args.output else Path(
        f"logs/vlp_grounding_format_report_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INFO] dataset_root={dataset_root}")
    print(f"[INFO] files={file_names}")
    print(f"[INFO] answer_type_counts={report['global_summary']['answer_type_counts']}")
    print(f"[INFO] prompt_class_counts={report['global_summary']['prompt_class_counts']}")
    print(f"[INFO] report_written_to={output_path.resolve()}")

    has_bbox = any(name == "bbox" for name, _ in report["global_summary"]["answer_type_counts"])
    has_point = any(name == "point" for name, _ in report["global_summary"]["answer_type_counts"])
    print(f"[INFO] bbox_answer_present={has_bbox}")
    print(f"[INFO] point_answer_present={has_point}")


if __name__ == "__main__":
    main()