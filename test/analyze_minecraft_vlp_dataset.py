#!/usr/bin/env python3
"""Analyze the composition and distribution of the minecraft-vlp dataset.

This script is designed for the Stage I / Stage II data bundle. It inspects the
top-level JSONL files under the dataset root and summarizes:

1. record counts per file
2. label hierarchy distributions
3. conversation structure statistics
4. multimodal image reference statistics
5. archive integrity for the corresponding images/*.zip file

The output is written as JSON so that it can be diffed across dataset snapshots.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_DATASET_ROOT = "/share/public_datasets/VLA/nitrogen/minecraft-vlp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze minecraft-vlp dataset composition.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory of minecraft-vlp.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output path. Defaults to ./logs/minecraft_vlp_report_<timestamp>.json",
    )
    parser.add_argument(
        "--max-records-per-file",
        type=int,
        default=0,
        help="If > 0, only analyze the first N records per JSONL file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="How many values to keep in the sorted top-k lists.",
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


def as_sorted_items(counter: Counter[str], top_k: int) -> list[list[Any]]:
    return [[key, int(count)] for key, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]]


def normalize_image_ref(image_ref: str) -> str:
    ref = image_ref.strip().replace("\\", "/")
    if ref.startswith("images/"):
        ref = ref[len("images/") :]
    return ref.lstrip("/")


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


def load_jsonl_records(jsonl_path: Path, max_records: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle):
            if max_records > 0 and row_idx >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                obj = {"_malformed_line": line}
            records.append(obj if isinstance(obj, dict) else {"_non_dict": obj})
    return records


def count_turns_and_roles(sample: dict[str, Any]) -> tuple[int, Counter[str], Counter[str]]:
    turns = sample.get("conversations")
    if not isinstance(turns, list):
        return 0, Counter(), Counter()

    role_counter: Counter[str] = Counter()
    content_type_counter: Counter[str] = Counter()
    valid_turns = 0
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        valid_turns += 1
        role = to_text(turn.get("role")).strip().lower() or "unknown"
        role_counter[role] += 1
        for item in normalize_content_items(turn.get("content")):
            item_type = to_text(item.get("type")).strip().lower() or "unknown"
            content_type_counter[item_type] += 1
    return valid_turns, role_counter, content_type_counter


def summarize_labels(label_value: Any) -> dict[str, Any]:
    labels = label_value if isinstance(label_value, list) else ([] if label_value is None else [label_value])
    label_text = [to_text(item).strip() for item in labels if to_text(item).strip()]
    return {
        "raw_label_count": len(labels),
        "label_values": label_text,
        "label_1": label_text[0] if len(label_text) >= 1 else "",
        "label_2": label_text[1] if len(label_text) >= 2 else "",
        "label_3": label_text[2] if len(label_text) >= 3 else "",
    }


def summarize_images(sample: dict[str, Any]) -> list[str]:
    image_field = sample.get("image")
    if isinstance(image_field, str):
        return [image_field]
    if isinstance(image_field, list):
        return [to_text(item) for item in image_field if to_text(item).strip()]
    return []


def inspect_archive(archive_path: Path) -> dict[str, Any]:
    if not archive_path.exists():
        return {"exists": False}

    with zipfile.ZipFile(archive_path, "r") as archive:
        names = archive.namelist()
        image_names = [name for name in names if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        prefix_counts: Counter[str] = Counter()
        for name in image_names:
            path = Path(name)
            prefix_counts[path.parent.as_posix()] += 1
        return {
            "exists": True,
            "member_count": len(names),
            "image_count": len(image_names),
            "top_level_dirs": as_sorted_items(prefix_counts, 20),
        }


def analyze_file(jsonl_path: Path, image_archive: Path, max_records: int, top_k: int) -> dict[str, Any]:
    records = load_jsonl_records(jsonl_path, max_records=max_records)
    file_report: dict[str, Any] = {
        "records": len(records),
        "archive": inspect_archive(image_archive),
        "schema_keys": Counter(),
        "label_1_counts": Counter(),
        "label_2_counts": Counter(),
        "label_3_counts": Counter(),
        "label_tuple_counts": Counter(),
        "turns_distribution": Counter(),
        "role_distribution": Counter(),
        "content_type_distribution": Counter(),
        "image_count_distribution": Counter(),
        "samples_with_image": 0,
        "samples_without_image": 0,
        "referenced_image_count": 0,
        "referenced_image_missing_in_archive": 0,
        "source_key_distribution": Counter(),
        "source_label_distribution": Counter(),
        "missing_label_rows": 0,
        "missing_conversation_rows": 0,
        "missing_image_rows": 0,
    }

    archive_members: set[str] = set()
    if image_archive.exists():
        with zipfile.ZipFile(image_archive, "r") as archive:
            archive_members = set(archive.namelist())

    for sample in records:
        file_report["schema_keys"].update(sample.keys())

        label_info = summarize_labels(sample.get("label"))
        if not label_info["label_values"]:
            file_report["missing_label_rows"] += 1
        else:
            file_report["label_1_counts"][label_info["label_1"]] += 1
            if label_info["label_2"]:
                file_report["label_2_counts"][label_info["label_2"]] += 1
            if label_info["label_3"]:
                file_report["label_3_counts"][label_info["label_3"]] += 1
            file_report["label_tuple_counts"][" | ".join(label_info["label_values"])] += 1

        turns = sample.get("conversations")
        if not isinstance(turns, list):
            file_report["missing_conversation_rows"] += 1
            turn_count = 0
        else:
            turn_count, roles, content_types = count_turns_and_roles(sample)
            file_report["role_distribution"].update(roles)
            file_report["content_type_distribution"].update(content_types)
        file_report["turns_distribution"][str(turn_count)] += 1

        image_refs = summarize_images(sample)
        if image_refs:
            file_report["samples_with_image"] += 1
        else:
            file_report["samples_without_image"] += 1
            file_report["missing_image_rows"] += 1

        file_report["image_count_distribution"][str(len(image_refs))] += 1
        for image_ref in image_refs:
            file_report["referenced_image_count"] += 1
            candidate = normalize_image_ref(image_ref)
            member_path = candidate if candidate.startswith(f"{jsonl_path.stem}/") else f"{jsonl_path.stem}/{candidate.split('/')[-1]}"
            if archive_members and member_path not in archive_members:
                file_report["referenced_image_missing_in_archive"] += 1

        source = sample.get("source")
        if isinstance(source, dict):
            file_report["source_key_distribution"].update(source.keys())
            if source.get("label") is not None:
                file_report["source_label_distribution"][to_text(source.get("label"))] += 1

    return {
        "jsonl": jsonl_path.name,
        "archive": image_archive.name,
        "records": file_report["records"],
        "archive_summary": file_report["archive"],
        "schema_keys": as_sorted_items(file_report["schema_keys"], top_k),
        "label_1_counts": as_sorted_items(file_report["label_1_counts"], top_k),
        "label_2_counts": as_sorted_items(file_report["label_2_counts"], top_k),
        "label_3_counts": as_sorted_items(file_report["label_3_counts"], top_k),
        "label_tuple_counts": as_sorted_items(file_report["label_tuple_counts"], top_k),
        "turns_distribution": as_sorted_items(file_report["turns_distribution"], top_k),
        "role_distribution": as_sorted_items(file_report["role_distribution"], top_k),
        "content_type_distribution": as_sorted_items(file_report["content_type_distribution"], top_k),
        "image_count_distribution": as_sorted_items(file_report["image_count_distribution"], top_k),
        "samples_with_image": file_report["samples_with_image"],
        "samples_without_image": file_report["samples_without_image"],
        "referenced_image_count": file_report["referenced_image_count"],
        "referenced_image_missing_in_archive": file_report["referenced_image_missing_in_archive"],
        "source_key_distribution": as_sorted_items(file_report["source_key_distribution"], top_k),
        "source_label_distribution": as_sorted_items(file_report["source_label_distribution"], top_k),
        "missing_label_rows": file_report["missing_label_rows"],
        "missing_conversation_rows": file_report["missing_conversation_rows"],
        "missing_image_rows": file_report["missing_image_rows"],
    }


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    image_root = dataset_root / "images"

    jsonl_files = sorted(dataset_root.glob("*.jsonl"))
    report: dict[str, Any] = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root),
        "image_root": str(image_root),
        "jsonl_files": [path.name for path in jsonl_files],
        "files": {},
        "global_summary": {},
    }

    global_record_count = 0
    global_label_1 = Counter()
    global_turns = Counter()
    global_roles = Counter()
    global_content_types = Counter()
    global_image_counts = Counter()

    for jsonl_path in jsonl_files:
        archive_path = image_root / f"{jsonl_path.stem}.zip"
        file_report = analyze_file(
            jsonl_path=jsonl_path,
            image_archive=archive_path,
            max_records=args.max_records_per_file,
            top_k=args.top_k,
        )
        report["files"][jsonl_path.name] = file_report

        global_record_count += int(file_report["records"])
        for key, count in file_report["label_1_counts"]:
            global_label_1[key] += int(count)
        for key, count in file_report["turns_distribution"]:
            global_turns[key] += int(count)
        for key, count in file_report["role_distribution"]:
            global_roles[key] += int(count)
        for key, count in file_report["content_type_distribution"]:
            global_content_types[key] += int(count)
        for key, count in file_report["image_count_distribution"]:
            global_image_counts[key] += int(count)

    report["global_summary"] = {
        "total_records": global_record_count,
        "label_1_counts": as_sorted_items(global_label_1, args.top_k),
        "turns_distribution": as_sorted_items(global_turns, args.top_k),
        "role_distribution": as_sorted_items(global_roles, args.top_k),
        "content_type_distribution": as_sorted_items(global_content_types, args.top_k),
        "image_count_distribution": as_sorted_items(global_image_counts, args.top_k),
    }

    output_path = Path(args.output) if args.output else Path(
        f"logs/minecraft_vlp_report_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INFO] Dataset root: {dataset_root}")
    print(f"[INFO] JSONL files: {', '.join(report['jsonl_files'])}")
    print(f"[INFO] Total records: {global_record_count}")
    print(f"[INFO] Report written to: {output_path.resolve()}")


if __name__ == "__main__":
    main()