#!/usr/bin/env python3
"""
Batch merge LoRA adapters back into base model.

Usage:
  python merge_lora_batch.py --list merges.txt

Each line in merges.txt should be (comma-separated, no spaces):
  /path/to/base,/path/to/adapter,/path/to/output

To use a custom delimiter, use --delimiter "\t" etc.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForVision2Seq


def parse_jobs(list_file: Path, delimiter: str) -> List[Tuple[str, str, str]]:
    """Parse the merge job list file."""
    jobs: List[Tuple[str, str, str]] = []
    with list_file.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter, skipinitialspace=True)
        for row in reader:
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            if row[0].lstrip().startswith("#"):
                continue
            if len(row) < 3:
                raise ValueError(
                    f"Row format error (expected 3 columns: base,adapter,output): {row}"
                )
            base_path, adapter_dir, output_dir = (cell.strip() for cell in row[:3])
            jobs.append((base_path, adapter_dir, output_dir))
    return jobs


def merge_one(base_model: str, adapter_dir: str, output_dir: str, dtype: str) -> None:
    """Merge a single LoRA adapter and save."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype)

    print(f"[Merge] base={base_model}, adapter={adapter_dir}, out={out_path}")
    base_vla = AutoModelForVision2Seq.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
    merged_vla = merged_vla.merge_and_unload()
    merged_vla.save_pretrained(out_path)
    print(f"[Done] saved to: {out_path}")

    # Try to also save tokenizer/processor; non-fatal if it fails.
    try:
        from transformers import AutoProcessor, AutoTokenizer  # lazy import

        AutoTokenizer.from_pretrained(base_model).save_pretrained(out_path)
        try:
            AutoProcessor.from_pretrained(base_model).save_pretrained(out_path)
        except Exception as proc_err:  # noqa: BLE001
            print(f"[Warn] processor not saved (can be ignored): {proc_err}")
    except Exception as tok_err:  # noqa: BLE001
        print(f"[Warn] tokenizer not saved (can be ignored): {tok_err}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Batch merge LoRA adapters into base model."
    )
    parser.add_argument(
        "--list",
        required=True,
        type=Path,
        help="Path to file containing base,adapter,output entries.",
    )
    parser.add_argument(
        "--delimiter",
        default=",",
        help="Delimiter for the list file (default: comma, use '\\t' for tabs).",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for loading (default: bfloat16).",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    jobs = parse_jobs(args.list, args.delimiter)
    if not jobs:
        raise SystemExit("No merge jobs found.")

    for base_model, adapter_dir, output_dir in jobs:
        merge_one(base_model, adapter_dir, output_dir, args.dtype)


if __name__ == "__main__":
    main()
