import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ONNX = ROOT_DIR / "evaluation" / "gaze360" / "checkpoint" / "ours_10.46.onnx"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX Runtime CPU inference FPS for the HFGAD gaze model."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_ONNX,
        help="Path to the ONNX model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Benchmark batch size.",
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
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Number of measured iterations.",
    )
    parser.add_argument(
        "--intra-op-threads",
        type=int,
        default=0,
        help="ONNX Runtime intra-op thread count. Use 0 for runtime default.",
    )
    parser.add_argument(
        "--inter-op-threads",
        type=int,
        default=0,
        help="ONNX Runtime inter-op thread count. Use 0 for runtime default.",
    )
    return parser


def create_session(model_path: Path, intra_op_threads: int, inter_op_threads: int):
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    options = ort.SessionOptions()
    if intra_op_threads > 0:
        options.intra_op_num_threads = intra_op_threads
    if inter_op_threads > 0:
        options.inter_op_num_threads = inter_op_threads

    session = ort.InferenceSession(
        model_path.as_posix(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    return session


def main():
    args = build_parser().parse_args()

    session = create_session(args.model, args.intra_op_threads, args.inter_op_threads)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    dummy_input = np.random.randn(args.batch_size, 3, args.height, args.width).astype(np.float32)

    for _ in range(args.warmup):
        session.run([output_name], {input_name: dummy_input})

    start = time.perf_counter()
    for _ in range(args.iters):
        output = session.run([output_name], {input_name: dummy_input})[0]
    end = time.perf_counter()

    total = end - start
    avg = total / args.iters
    fps = args.batch_size / avg

    print(f"Model: {args.model}")
    print("Provider: CPUExecutionProvider")
    print(f"Input shape: {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Measured iterations: {args.iters}")
    print(f"Total seconds: {total:.6f}")
    print(f"Average seconds: {avg:.6f}")
    print(f"FPS: {fps:.2f}")
    print(f"Intra-op threads: {args.intra_op_threads or 'default'}")
    print(f"Inter-op threads: {args.inter_op_threads or 'default'}")


if __name__ == "__main__":
    main()
