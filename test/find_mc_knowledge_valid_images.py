#!/usr/bin/env python3
"""Find the image files referenced by mc-knowledge-valid.jsonl across all VLP zip archives.

The mc-knowledge-valid subset has image references but no dedicated zip archive in the
current public layout. This script scans all other archives under minecraft-vlp/images
and reports where each referenced image can be found.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DATASET_ROOT = "/share/public_datasets/VLA/nitrogen/minecraft-vlp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search all VLP archives for mc-knowledge-valid images.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory of minecraft-vlp.",
    )
    parser.add_argument(
        "--target-jsonl",
        type=str,
        default="mc-knowledge-valid.jsonl",
        help="Target JSONL file relative to dataset root.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional JSON output path for the search report.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on how many image references to search. 0 means all.",
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


def normalize_image_ref(image_ref: str) -> tuple[str, str]:
    """Return (expected_member_path, basename).

    The JSONL usually stores refs like:
        images/mc-grounding-point-gui/<file>.jpg

    The zip members are typically:
        mc-grounding-point-gui/<file>.jpg
    """

    ref = image_ref.strip().replace("\\", "/")
    if ref.startswith("images/"):
        ref = ref[len("images/") :]
    ref = ref.lstrip("/")
    return ref, Path(ref).name


def load_target_image_refs(target_jsonl: Path, limit: int) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    with target_jsonl.open("r", encoding="utf-8") as handle:
        for sample_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if not isinstance(sample, dict):
                continue

            image_field = sample.get("image")
            if isinstance(image_field, str):
                image_items = [image_field]
            elif isinstance(image_field, list):
                image_items = [to_text(item) for item in image_field if to_text(item).strip()]
            else:
                image_items = []

            for image_ref in image_items:
                expected_member, basename = normalize_image_ref(image_ref)
                refs.append(
                    {
                        "sample_index": sample_index,
                        "sample_id": sample.get("id"),
                        "image_ref": image_ref,
                        "expected_member": expected_member,
                        "basename": basename,
                    }
                )
                if limit > 0 and len(refs) >= limit:
                    return refs
    return refs


def build_archive_indexes(archive_paths: list[Path]) -> dict[str, dict[str, list[str]]]:
    indexes: dict[str, dict[str, list[str]]] = {}
    for archive_path in archive_paths:
        member_by_name: dict[str, list[str]] = defaultdict(list)
        member_by_basename: dict[str, list[str]] = defaultdict(list)
        with zipfile.ZipFile(archive_path, "r") as archive:
            for member in archive.namelist():
                if member.endswith("/"):
                    continue
                member_by_name[member].append(member)
                member_by_basename[Path(member).name].append(member)
        indexes[archive_path.name] = {
            "member_count": len(member_by_name),
            "members": member_by_name,
            "basenames": member_by_basename,
        }
    return indexes


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    target_jsonl = dataset_root / args.target_jsonl
    image_dir = dataset_root / "images"

    if not target_jsonl.exists():
        raise SystemExit(f"Target JSONL not found: {target_jsonl}")
    if not image_dir.exists():
        raise SystemExit(f"Image directory not found: {image_dir}")

    archive_paths = sorted(image_dir.glob("*.zip"))
    if not archive_paths:
        raise SystemExit(f"No zip archives found under {image_dir}")

    refs = load_target_image_refs(target_jsonl, limit=args.limit)
    archive_indexes = build_archive_indexes(archive_paths)

    by_archive_counter: Counter[str] = Counter()
    missing_refs = []
    ambiguous_refs = []
    matched_refs = []

    for ref in refs:
        exact_hits = []
        basename_hits = []
        for archive_name, index in archive_indexes.items():
            if ref["expected_member"] in index["members"]:
                exact_hits.extend((archive_name, member) for member in index["members"][ref["expected_member"]])
            if ref["basename"] in index["basenames"]:
                basename_hits.extend((archive_name, member) for member in index["basenames"][ref["basename"]])

        hits = exact_hits if exact_hits else basename_hits
        unique_archives = sorted({archive_name for archive_name, _ in hits})

        if hits:
            matched_refs.append(
                {
                    **ref,
                    "match_type": "exact" if exact_hits else "basename",
                    "matches": [{"archive": archive, "member": member} for archive, member in hits],
                }
            )
            for archive_name in unique_archives:
                by_archive_counter[archive_name] += 1
            if len(hits) > 1:
                ambiguous_refs.append(
                    {
                        **ref,
                        "match_type": "exact" if exact_hits else "basename",
                        "matches": [{"archive": archive, "member": member} for archive, member in hits],
                    }
                )
        else:
            missing_refs.append(ref)

    report = {
        "dataset_root": str(dataset_root),
        "target_jsonl": args.target_jsonl,
        "num_archives": len(archive_paths),
        "archives": [path.name for path in archive_paths],
        "target_image_ref_count": len(refs),
        "matched_image_ref_count": len(matched_refs),
        "missing_image_ref_count": len(missing_refs),
        "ambiguous_image_ref_count": len(ambiguous_refs),
        "matches_by_archive": [[name, int(count)] for name, count in by_archive_counter.most_common()],
        "matched_refs": matched_refs,
        "missing_refs": missing_refs,
        "ambiguous_refs": ambiguous_refs,
    }

    print(f"[INFO] dataset_root={dataset_root}")
    print(f"[INFO] target_jsonl={args.target_jsonl}")
    print(f"[INFO] archives={len(archive_paths)}")
    print(f"[INFO] target_image_ref_count={len(refs)}")
    print(f"[INFO] matched_image_ref_count={len(matched_refs)}")
    print(f"[INFO] missing_image_ref_count={len(missing_refs)}")
    print(f"[INFO] ambiguous_image_ref_count={len(ambiguous_refs)}")
    print("[INFO] matches_by_archive:")
    for archive_name, count in report["matches_by_archive"]:
        print(f"  - {archive_name}: {count}")

    if missing_refs:
        print("[WARN] missing references:")
        for item in missing_refs[:20]:
            print(f"  - sample_index={item['sample_index']} id={item['sample_id']} ref={item['image_ref']}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] wrote report to: {output_path.resolve()}")


if __name__ == "__main__":
    main()