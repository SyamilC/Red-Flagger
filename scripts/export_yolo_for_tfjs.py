"""Export the cached YOLOv8 model to TensorFlow.js assets for the Chrome extension.

This helper depends on Ultralytics export support plus TensorFlow/TensorFlow.js converter
packages. It writes to chrome_extension/models/yolo_model_tfjs by default.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "models" / "yolo_model.pt"
DEFAULT_OUTPUT = ROOT / "chrome_extension" / "models" / "yolo_model_tfjs"


def export_yolo(weights: Path, output_dir: Path, imgsz: int, overwrite: bool) -> None:
    if not weights.exists() or weights.stat().st_size == 0:
        raise FileNotFoundError(f"YOLO weights were not found or are empty: {weights}")

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)

    model = YOLO(str(weights))
    exported = Path(model.export(format="tfjs", imgsz=imgsz))
    if not exported.exists():
        raise RuntimeError(f"Ultralytics reported an export path that does not exist: {exported}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if exported.is_dir():
        shutil.copytree(exported, output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exported, output_dir / exported.name)

    print(f"Wrote {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLOv8 weights to TensorFlow.js assets.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    export_yolo(args.weights, args.output, args.imgsz, args.overwrite)


if __name__ == "__main__":
    main()