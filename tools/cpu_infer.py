import argparse
from pathlib import Path
import sys

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.Gaze import GazeModel

DEFAULT_CHECKPOINT = ROOT_DIR / "evaluation" / "gaze360" / "checkpoint" / "ours_10.46.pt"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Load the HFGAD gaze model on CPU and run a smoke-test forward pass."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for the dummy input.",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible dummy input.",
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

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    model = load_model(args.checkpoint).to(device)
    dummy_input = torch.randn(args.batch_size, 3, args.height, args.width, device=device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Input shape: {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print("Output sample:")
    print(output)


if __name__ == "__main__":
    main()
