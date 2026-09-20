import argparse
from pathlib import Path
import sys

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.Gaze import GazeModel

DEFAULT_CHECKPOINT = ROOT_DIR / "evaluation" / "gaze360" / "checkpoint" / "ours_10.46.pt"
DEFAULT_ONNX = ROOT_DIR / "evaluation" / "gaze360" / "checkpoint" / "ours_10.46.onnx"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Export the HFGAD gaze checkpoint to ONNX for CPU inference."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the PyTorch checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ONNX,
        help="Path to save the ONNX model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Dummy input batch size used for export.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=224,
        help="Input image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=224,
        help="Input image width.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    return parser


def load_model(checkpoint_path: Path) -> GazeModel:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model = GazeModel(backbone="resnet18")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def main():
    args = build_parser().parse_args()

    model = load_model(args.checkpoint)
    dummy_input = torch.randn(args.batch_size, 3, args.height, args.width)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_input,
            args.output.as_posix(),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["gaze"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "gaze": {0: "batch_size"},
            },
        )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"ONNX saved to: {args.output}")
    print(f"Input shape: {tuple(dummy_input.shape)}")
    print(f"Opset: {args.opset}")


if __name__ == "__main__":
    main()
