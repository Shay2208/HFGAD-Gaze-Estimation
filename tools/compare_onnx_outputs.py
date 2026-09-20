import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FP32 = ROOT_DIR / "evaluation" / "gaze360" / "checkpoint" / "ours_10.46.onnx"
DEFAULT_INT8 = ROOT_DIR / "evaluation" / "gaze360" / "checkpoint" / "ours_10.46.int8.onnx"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare outputs between FP32 and INT8 ONNX models on the same random input."
    )
    parser.add_argument("--fp32", type=Path, default=DEFAULT_FP32)
    parser.add_argument("--int8", type=Path, default=DEFAULT_INT8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def run_model(model_path: Path, x: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(
        model_path.as_posix(),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session.run([output_name], {input_name: x})[0]


def main():
    args = build_parser().parse_args()

    if not args.fp32.exists():
        raise FileNotFoundError(f"FP32 ONNX model not found: {args.fp32}")
    if not args.int8.exists():
        raise FileNotFoundError(f"INT8 ONNX model not found: {args.int8}")

    rng = np.random.default_rng(args.seed)
    x = rng.standard_normal((args.batch_size, 3, args.height, args.width), dtype=np.float32)

    y_fp32 = run_model(args.fp32, x)
    y_int8 = run_model(args.int8, x)

    diff = np.abs(y_fp32 - y_int8)
    print(f"FP32 output shape: {tuple(y_fp32.shape)}")
    print(f"INT8 output shape: {tuple(y_int8.shape)}")
    print(f"Mean abs diff: {diff.mean():.8f}")
    print(f"Max abs diff: {diff.max():.8f}")
    print("FP32 sample:")
    print(y_fp32)
    print("INT8 sample:")
    print(y_int8)


if __name__ == "__main__":
    main()
