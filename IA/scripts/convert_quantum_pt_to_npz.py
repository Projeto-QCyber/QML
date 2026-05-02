#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


ARRAY_KEYS = ("weights", "w_ro", "bias", "alpha", "score_sign")


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def convert_pt_to_npz(pt_path: Path, out_prefix: Path) -> None:
    import torch

    state = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError("Expected the .pt file to contain a dict-like quantum model state.")

    arrays: Dict[str, np.ndarray] = {}
    for key in ARRAY_KEYS:
        if key not in state:
            raise ValueError(f"Missing required tensor key '{key}' in '{pt_path}'.")
        arrays[key] = _to_numpy(state[key])

    metadata = {
        str(k): _to_jsonable(v)
        for k, v in state.items()
        if k not in ARRAY_KEYS
    }
    metadata["artifact_format"] = "qcyber_quantum_npz"
    metadata["array_file"] = out_prefix.with_suffix(".npz").name
    metadata["source_pt_file"] = pt_path.name

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_prefix.with_suffix(".npz"), **arrays)
    with out_prefix.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch .pt quantum checkpoint into lightweight .npz/.json artifacts.",
    )
    parser.add_argument("input", help="Path to the source .pt checkpoint.")
    parser.add_argument(
        "--out-prefix",
        help="Output path without extension. Defaults to the input path stem.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pt_path = Path(args.input)
    out_prefix = Path(args.out_prefix) if args.out_prefix else pt_path.with_suffix("")
    convert_pt_to_npz(pt_path, out_prefix)
    print(f"Saved {out_prefix.with_suffix('.npz')}")
    print(f"Saved {out_prefix.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
