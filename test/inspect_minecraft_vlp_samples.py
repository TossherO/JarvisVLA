#!/usr/bin/env python3
"""Inspect a few representative minecraft-vlp samples.

This script is meant for qualitative review. It prints the exact conversation
layout, label hierarchy, image references, and source metadata for a small set
of randomly selected or sequentially selected samples.
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path
from typing import Any


DEFAULT_DATASET_ROOT = "/share/public_datasets/VLA/nitrogen/minecraft-vlp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect minecraft-vlp samples.")
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
        help="Inspect only specific JSONL files. Can be passed multiple times.",
    )
    parser.add_argument(
        "--num-samples-per-file",
        type=int,
        default=5,
        help="How many samples to inspect per file.",
    )
    parser.add_argument(
        "--random-sample",
        action="store_true",
        help="Randomly sample records instead of taking the first N.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=43,
        help="Random seed used for sampling.",
    )
    parser.add_argument(
        "--text-max-chars",
        type=int,
        default=220,
        help="Maximum characters to show for each text field.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output path for a JSON inspection report.",
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


def shorten(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


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


def summarize_grounding_content(item: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "type": to_text(item.get("type")).strip() or "unknown",
        "text": shorten(to_text(item.get("text")), 220),
    }
    if "point" in item:
        summary["point"] = item.get("point")
    if "bbox" in item:
        summary["bbox"] = item.get("bbox")
    if "label" in item:
        summary["label"] = item.get("label")
    return summary


def summarize_source(source: Any) -> dict[str, Any] | None:
    if not isinstance(source, dict):
        return None
    summary: dict[str, Any] = {"keys": sorted(source.keys())}
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


def load_records(jsonl_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
    return records


def choose_indices(total: int, k: int, random_sample: bool, seed: int) -> list[int]:
    if total <= 0 or k <= 0:
        return []
    count = min(total, k)
    if random_sample:
        rng = random.Random(seed)
        return sorted(rng.sample(range(total), count))
    return list(range(count))


def inspect_sample(sample: dict[str, Any], index: int, text_max_chars: int, archive_members: set[str], jsonl_stem: str) -> dict[str, Any]:
    labels = sample.get("label") if isinstance(sample.get("label"), list) else ([] if sample.get("label") is None else [sample.get("label")])
    label_values = [to_text(item).strip() for item in labels if to_text(item).strip()]

    image_field = sample.get("image")
    if isinstance(image_field, str):
        image_refs = [image_field]
    elif isinstance(image_field, list):
        image_refs = [to_text(item) for item in image_field if to_text(item).strip()]
    else:
        image_refs = []

    image_members = []
    for ref in image_refs:
        candidate = ref.strip().replace("\\", "/")
        if candidate.startswith("images/"):
            candidate = candidate[len("images/") :]
        candidate = candidate.lstrip("/")
        member = candidate if candidate.startswith(f"{jsonl_stem}/") else f"{jsonl_stem}/{candidate.split('/')[-1]}"
        image_members.append({"ref": ref, "archive_member": member, "exists_in_archive": member in archive_members})

    turns = []
    conversations = sample.get("conversations")
    if isinstance(conversations, list):
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            content_items = []
            for item in normalize_content_items(turn.get("content")):
                content_items.append(summarize_grounding_content(item))
            turns.append(
                {
                    "role": to_text(turn.get("role")).strip() or "unknown",
                    "num_content_items": len(content_items),
                    "content": content_items,
                }
            )

    return {
        "index": index,
        "id": sample.get("id"),
        "label": label_values,
        "source": summarize_source(sample.get("source")),
        "image_refs": image_members,
        "turns": turns,
    }


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    jsonl_files = sorted(dataset_root.glob("*.jsonl"))
    if args.file:
        wanted = {name for name in args.file}
        jsonl_files = [path for path in jsonl_files if path.name in wanted]

    if not jsonl_files:
        raise SystemExit(f"No JSONL files found under {dataset_root}")

    report: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "files": {},
    }

    for jsonl_path in jsonl_files:
        archive_path = dataset_root / "images" / f"{jsonl_path.stem}.zip"
        archive_members: set[str] = set()
        if archive_path.exists():
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive_members = set(archive.namelist())

        records = load_records(jsonl_path)
        indices = choose_indices(len(records), args.num_samples_per_file, args.random_sample, args.seed)
        print("=" * 100)
        print(f"FILE: {jsonl_path.name}")
        print(f"records={len(records)} archive_exists={archive_path.exists()} archive={archive_path.name}")
        print(f"selected_indices={indices}")

        inspected = []
        for idx in indices:
            sample = records[idx]
            info = inspect_sample(sample, idx, args.text_max_chars, archive_members, jsonl_path.stem)
            inspected.append(info)
            print("-" * 100)
            print(f"index={info['index']} id={info['id']}")
            print(f"label={info['label']}")
            if info.get("source") is not None:
                print(f"source={info['source']}")
            print("image_refs:")
            for ref in info["image_refs"]:
                print(f"  - {ref['ref']} -> {ref['archive_member']} exists={ref['exists_in_archive']}")
            print(f"turns={len(info['turns'])}")
            for turn_idx, turn in enumerate(info["turns"]):
                print(f"  turn[{turn_idx}] role={turn['role']} content_items={turn['num_content_items']}")
                for item_idx, content in enumerate(turn["content"]):
                    print(f"    content[{item_idx}] type={content['type']} text={content['text']}")
                    if "point" in content:
                        print(f"      point={content['point']}")
                    if "bbox" in content:
                        print(f"      bbox={content['bbox']}")
                    if "label" in content:
                        print(f"      label={content['label']}")

        report["files"][jsonl_path.name] = {
            "records": len(records),
            "archive_exists": archive_path.exists(),
            "archive": archive_path.name,
            "selected_indices": indices,
            "samples": inspected,
        }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] Wrote inspection report to: {output_path.resolve()}")


if __name__ == "__main__":
    main()