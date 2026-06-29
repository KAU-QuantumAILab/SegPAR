#!/usr/bin/env python3
"""Reproduce shared segmentation sample sets for third-party users.

This script has one fixed behavior:
1) Read fixed manifest files for ADE20K, Cityscapes, and VOC2012.
2) Copy listed files from each full dataset root.
3) Write them to fixed sample output directories while preserving relative paths.
"""

from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_DATASET_ROOT = Path("/workspace/Dataset")
PROGRESS_EVERY = 200

JOBS = [
    {
        "name": "ADE20K",
        "source_root": ORIGINAL_DATASET_ROOT / "ADE20K",
        "manifest_path": PROJECT_ROOT / "manifests" / "ade20k_200_manifest.txt",
        "dest_root": PROJECT_ROOT / "datasets" / "ade20k",
    },
    {
        "name": "Cityscapes",
        "source_root": ORIGINAL_DATASET_ROOT / "cityscapes",
        "manifest_path": PROJECT_ROOT / "manifests" / "cityscapes_300_manifest.txt",
        "dest_root": PROJECT_ROOT / "datasets" / "cityscapes",
    },
    {
        "name": "VOC2012",
        "source_root": ORIGINAL_DATASET_ROOT / "VOCdevkit" / "VOC2012",
        "manifest_path": PROJECT_ROOT / "manifests" / "voc2012_200_manifest.txt",
        "dest_root": PROJECT_ROOT / "datasets" / "VOC2012",
    },
]


def load_manifest(manifest_path: Path) -> list[Path]:
    """Load relative paths from a manifest file."""
    if not manifest_path.exists() or not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    rel_paths: list[Path] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            rel_path = Path(line)
            if rel_path.is_absolute():
                raise ValueError(
                    f"Manifest line {line_no} must be a relative path: {line}"
                )
            if ".." in rel_path.parts:
                raise ValueError(
                    f"Manifest line {line_no} contains invalid '..' segment: {line}"
                )
            rel_paths.append(rel_path)

    # Keep first occurrence order while removing duplicates.
    return list(dict.fromkeys(rel_paths))


def validate_source_files(src_root: Path, rel_paths: list[Path]) -> None:
    """Fail fast if any manifest path is missing in source dataset."""
    if not src_root.exists() or not src_root.is_dir():
        raise FileNotFoundError(f"Source dataset directory not found: {src_root}")

    missing: list[Path] = []
    for rel_path in rel_paths:
        src_file = src_root / rel_path
        if not src_file.exists() or not src_file.is_file():
            missing.append(rel_path)

    if missing:
        preview = "\n".join(f"  - {p.as_posix()}" for p in missing[:10])
        raise FileNotFoundError(
            "Some manifest files are missing in source dataset.\n"
            f"Missing count: {len(missing)}\n"
            f"First missing paths:\n{preview}"
        )


def reproduce_samples(name: str, src_root: Path, manifest_path: Path, dst_root: Path) -> None:
    """Copy one dataset sample set from source to destination using manifest."""
    rel_paths = load_manifest(manifest_path)
    if not rel_paths:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")

    validate_source_files(src_root, rel_paths)

    dst_root.mkdir(parents=True, exist_ok=True)

    total = len(rel_paths)
    print(f"[{name}] Source      : {src_root}")
    print(f"[{name}] Manifest    : {manifest_path}")
    print(f"[{name}] Destination : {dst_root}")
    print(f"[{name}] Files to copy: {total}")

    for index, rel_path in enumerate(rel_paths, start=1):
        src_file = src_root / rel_path
        dst_file = dst_root / rel_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        # Overwrite destination to guarantee deterministic reproduction.
        shutil.copy2(src_file, dst_file)

        if index % PROGRESS_EVERY == 0 or index == total:
            print(f"[{name}] [{index}/{total}] copied")

    print(f"[{name}] Done. Reproduced files: {total}\n")


def main() -> None:
    """Entry point."""
    for job in JOBS:
        reproduce_samples(
            name=job["name"],
            src_root=job["source_root"],
            manifest_path=job["manifest_path"],
            dst_root=job["dest_root"],
        )


if __name__ == "__main__":
    main()
