from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch


def _allowlist_sklearn() -> Optional[type]:
    try:
        from sklearn.preprocessing import MinMaxScaler as SkMinMax
        import torch.serialization as ts

        ts.add_safe_globals([SkMinMax])
        try:
            import sklearn.preprocessing._data as sk_data
            ts.add_safe_globals([getattr(sk_data, "MinMaxScaler", SkMinMax)])
        except Exception:
            pass
        return SkMinMax
    except Exception:
        return None


def _rebuild_minmax(scaler_state: Dict[str, Any], SkMinMax: type) -> Any:
    sc = SkMinMax(feature_range=tuple(scaler_state.get("feature_range", (0, 1))))
    for attr in ["min_", "scale_", "data_min_", "data_max_", "data_range_", "n_samples_seen_"]:
        val = scaler_state.get(attr)
        if val is not None:
            setattr(sc, attr, np.asarray(val))
    return sc


def _load_state(path: Path) -> Dict[str, Any]:
    return torch.load(str(path), map_location="cpu", weights_only=False)


def _save_state(state: Dict[str, Any], out_path: Path) -> None:
    torch.save(state, str(out_path))


def _iter_paths(input_path: Path, pattern: str) -> Iterable[Path]:
    if input_path.is_dir():
        yield from sorted(input_path.glob(pattern))
    else:
        yield input_path


def _sklearn_tag() -> str:
    try:
        import sklearn  # type: ignore
        ver = getattr(sklearn, "__version__", "unknown")
    except Exception:
        return "sklearn"

    parts = [p for p in ver.split(".") if p.isdigit()]
    while len(parts) < 3:
        parts.append("0")
    return "sklearn" + "".join(parts[:3])


def convert_checkpoint(
    in_path: Path,
    out_path: Path,
    rebuild_scaler: bool,
    SkMinMax: Optional[type],
) -> None:
    state = _load_state(in_path)

    scaler = state.get("scaler")
    scaler_state = state.get("scaler_state")

    if rebuild_scaler and scaler_state is not None and SkMinMax is not None:
        scaler = _rebuild_minmax(scaler_state, SkMinMax)
        state["scaler"] = scaler

    _save_state(state, out_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-save .pt checkpoints with the current scikit-learn version.",
    )
    parser.add_argument("input", help="Path to a .pt file or a directory of .pt files.")
    parser.add_argument(
        "--out",
        help="Output file or directory. If omitted, writes alongside input with '.sklearn170.pt' suffix.",
    )
    parser.add_argument(
        "--pattern",
        default="*.pt",
        help="Glob pattern for directory input (default: *.pt).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input files in-place.",
    )
    parser.add_argument(
        "--rebuild-scaler",
        action="store_true",
        help="Rebuild MinMaxScaler from scaler_state before saving.",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    out_path = Path(args.out) if args.out else None

    SkMinMax = _allowlist_sklearn()
    tag = _sklearn_tag()

    if input_path.is_dir():
        if out_path is None and not args.in_place:
            out_path = input_path
        if out_path is None:
            out_path = input_path
        out_path.mkdir(parents=True, exist_ok=True)

        for pt in _iter_paths(input_path, args.pattern):
            if args.in_place:
                dest = pt
            else:
                dest = out_path / f"{pt.stem}.{tag}.pt"
            convert_checkpoint(pt, dest, args.rebuild_scaler, SkMinMax)
    else:
        if args.in_place:
            dest = input_path
        else:
            dest = out_path if out_path else input_path.with_name(f"{input_path.stem}.{tag}.pt")
        convert_checkpoint(input_path, dest, args.rebuild_scaler, SkMinMax)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
