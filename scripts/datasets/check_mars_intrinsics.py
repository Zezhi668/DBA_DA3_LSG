#!/usr/bin/env python3
"""Check MARS-LIVG TUM/config/manifest intrinsic consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that a runtime YAML and/or LiDAR-depth training manifest "
            "use the intrinsics written by the TUM exporter."
        )
    )
    parser.add_argument("paths", nargs="+", type=Path, help="Config YAMLs or manifest.jsonl files.")
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--manifest-samples", type=int, default=8)
    return parser.parse_args()


def load_tum_intrinsics(tum_dir: Path) -> tuple[list[float], list[int], str]:
    export_info = tum_dir / "export_info.json"
    if export_info.is_file():
        data = json.loads(export_info.read_text(encoding="utf-8"))
        return (
            [float(v) for v in data["calibration_txt_fx_fy_cx_cy"]],
            [int(v) for v in data["written_resolution"]],
            str(export_info),
        )

    camera_info = tum_dir / "camera_info.yaml"
    if camera_info.is_file():
        data = yaml.safe_load(camera_info.read_text(encoding="utf-8"))
        return (
            [float(v) for v in data["calibration_txt_fx_fy_cx_cy"]],
            [int(v) for v in data["written_resolution"]],
            str(camera_info),
        )

    calibration = tum_dir / "calibration.txt"
    if calibration.is_file():
        values = [float(v) for v in calibration.read_text(encoding="utf-8").split()[:4]]
        if len(values) != 4:
            raise ValueError(f"{calibration} does not contain fx fy cx cy")
        return values, [], str(calibration)

    raise FileNotFoundError(
        f"No export_info.json, camera_info.yaml, or calibration.txt found in {tum_dir}"
    )


def expected_runtime_intrinsics(fx_fy_cx_cy: Iterable[float]) -> dict[str, float]:
    fx, fy, cx, cy = [float(v) for v in fx_fy_cx_cy]
    # The DPT-LSG TUM loader expects this transposed naming convention.
    return {"fu": fy, "fv": fx, "cu": cy, "cv": cx}


def max_abs_diff(a: Iterable[float], b: Iterable[float]) -> float:
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def status(ok: bool) -> str:
    return "OK" if ok else "FAIL"


def check_config(path: Path, tol: float) -> bool:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    tum_dir = Path(cfg["dataset"]["root"])
    print(f"\nCONFIG {path}")
    print(f"  tum_dir: {tum_dir}")
    if not tum_dir.is_dir():
        print("  FAIL: TUM directory does not exist")
        return False

    fx_fy_cx_cy, written_wh, source = load_tum_intrinsics(tum_dir)
    expected = expected_runtime_intrinsics(fx_fy_cx_cy)
    actual = cfg["intrinsic"]

    intr_keys = ["fu", "fv", "cu", "cv"]
    diff = max_abs_diff([actual[k] for k in intr_keys], [expected[k] for k in intr_keys])
    intr_ok = diff <= tol

    wh_ok = True
    if written_wh:
        width, height = written_wh
        wh_ok = int(actual["W"]) == width and int(actual["H"]) == height

    print(f"  source: {source}")
    print(f"  export fx fy cx cy: {fx_fy_cx_cy}")
    print(f"  expected fu fv cu cv: {[expected[k] for k in intr_keys]}")
    print(f"  yaml fu fv cu cv: {[actual[k] for k in intr_keys]}")
    print(f"  intrinsics: {status(intr_ok)} max_abs_diff={diff:.6g}")
    if written_wh:
        print(f"  resolution: {status(wh_ok)} yaml W,H={[actual['W'], actual['H']]} export W,H={written_wh}")
    return intr_ok and wh_ok


def infer_tum_dir_from_manifest(manifest: Path, first_record: dict[str, object]) -> Path:
    image_path = Path(str(first_record["image"]))
    for parent in image_path.parents:
        if (parent / "rgb.txt").is_file():
            return parent
    # Normal layout is <tum_dir>/rgb/<frame>.
    return image_path.parent.parent


def sample_manifest_records(manifest: Path, max_samples: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
                if len(records) >= max_samples:
                    break
    if not records:
        raise ValueError(f"No records in {manifest}")
    return records


def check_manifest(path: Path, tol: float, samples: int) -> bool:
    records = sample_manifest_records(path, samples)
    tum_dir = infer_tum_dir_from_manifest(path, records[0])
    print(f"\nMANIFEST {path}")
    print(f"  inferred_tum_dir: {tum_dir}")

    fx_fy_cx_cy, written_wh, source = load_tum_intrinsics(tum_dir)
    manifest_intrinsics = [records[0].get("intrinsics_fx_fy_cx_cy") for _ in records[:1]][0]
    if manifest_intrinsics is None:
        print("  FAIL: manifest records do not contain intrinsics_fx_fy_cx_cy")
        return False

    intr_ok = True
    wh_ok = True
    for record in records:
        record_intrinsics = [float(v) for v in record["intrinsics_fx_fy_cx_cy"]]
        intr_ok = intr_ok and max_abs_diff(record_intrinsics, fx_fy_cx_cy) <= tol
        if written_wh:
            width, height = written_wh
            wh_ok = wh_ok and int(record["width"]) == width and int(record["height"]) == height

    diff = max_abs_diff([float(v) for v in manifest_intrinsics], fx_fy_cx_cy)
    print(f"  source: {source}")
    print(f"  export fx fy cx cy: {fx_fy_cx_cy}")
    print(f"  manifest fx fy cx cy: {manifest_intrinsics}")
    print(f"  intrinsics: {status(intr_ok)} max_abs_diff_first={diff:.6g} samples={len(records)}")
    if written_wh:
        print(
            "  resolution: "
            f"{status(wh_ok)} manifest W,H={[records[0]['width'], records[0]['height']]} "
            f"export W,H={written_wh}"
        )
    return intr_ok and wh_ok


def main() -> None:
    args = parse_args()
    all_ok = True
    for path in args.paths:
        if path.name.endswith(".jsonl"):
            ok = check_manifest(path, args.tol, args.manifest_samples)
        else:
            ok = check_config(path, args.tol)
        all_ok = all_ok and ok
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
