import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ONNX = ROOT_DIR / "evaluation" / "gaze360" / "checkpoint" / "ours_10.46.onnx"
DEFAULT_INT8_ONNX = ROOT_DIR / "evaluation" / "gaze360" / "checkpoint" / "ours_10.46.int8.onnx"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Apply dynamic INT8 quantization to the exported ONNX model."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_ONNX,
        help="Path to the FP32 ONNX model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_INT8_ONNX,
        help="Path to save the INT8 ONNX model.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"ONNX model not found: {args.model}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=args.model.as_posix(),
        model_output=args.output.as_posix(),
        weight_type=QuantType.QInt8,
    )

    print(f"FP32 model: {args.model}")
    print(f"INT8 model: {args.output}")


if __name__ == "__main__":
    main()
