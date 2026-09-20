import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent


def check_imports():
    print("[1/4] 检查依赖...")
    missing = []
    for mod in ["torch", "cv2", "numpy", "onnxruntime", "yaml", "easydict"]:
        try:
            __import__(mod)
            print(f"  [OK] {mod}")
        except ImportError as e:
            print(f"  [MISSING] {mod}: {e}")
            missing.append(mod)
    if missing:
        print(f"\n缺少依赖：{missing}，请运行 pip install -r requirements.txt")
        return False
    return True


def check_onnx_model():
    print("[2/4] 检查 ONNX 模型文件...")
    model_path = ROOT_DIR / "evaluation" / "gaze360" / "checkpoint" / "ours_10.46.onnx"
    if not model_path.exists():
        print(f"  [MISSING] {model_path}")
        return False
    print(f"  [OK] {model_path.name} ({model_path.stat().st_size / 1e6:.1f} MB)")
    return True


def check_cascade():
    print("[3/4] 检查人脸检测级联文件...")
    try:
        import cv2
        cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if cascade.exists():
            print(f"  [OK] {cascade.name}")
            return True
        print(f"  [MISSING] {cascade}")
    except Exception as e:
        print(f"  [ERROR] {e}")
    return False


def run_inference():
    print("[4/4] 运行 ONNX 烟雾推理...")
    try:
        import numpy as np
        import onnxruntime as ort
        model_path = ROOT_DIR / "evaluation" / "gaze360" / "checkpoint" / "ours_10.46.onnx"
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        shape = sess.get_inputs()[0].shape
        h = shape[2] if isinstance(shape[2], int) else 224
        w = shape[3] if isinstance(shape[3], int) else 224
        dummy = np.random.randn(1, 3, h, w).astype(np.float32)
        out = sess.run(None, {inp_name: dummy})
        print(f"  [OK] 输入形状 {dummy.shape} -> 输出形状 {out[0].shape}")
        return True
    except Exception as e:
        print(f"  [ERROR] 推理失败：{e}")
        return False


def main():
    print("=" * 50)
    print("HFGAD 环境验证脚本")
    print("=" * 50)
    ok = True
    ok &= check_imports()
    ok &= check_onnx_model()
    ok &= check_cascade()
    if ok:
        ok &= run_inference()
    print("=" * 50)
    if ok:
        print("全部通过！可以运行 python webcam_gaze_demo.py")
        sys.exit(0)
    else:
        print("存在未通过项，请按上方提示修复。")
        sys.exit(1)


if __name__ == "__main__":
    main()
