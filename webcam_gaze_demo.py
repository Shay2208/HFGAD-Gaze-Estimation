import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT_DIR / "evaluation" / "gaze360" / "checkpoint" / "ours_10.46.onnx"
# DEFAULT_MODEL = ROOT_DIR / "evaluation" / "mpii" / "checkpoint" / "p00.label" / "hfgad.onnx"
DEFAULT_CASCADE = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"


class PerfAverager:
    def __init__(self, momentum: float = 0.9):
        self.momentum = momentum
        self.values = {}

    def update(self, key: str, value: float):
        if key in self.values:
            self.values[key] = self.momentum * self.values[key] + (1 - self.momentum) * value
        else:
            self.values[key] = value

    def get_ms(self, key: str) -> float:
        return self.values.get(key, 0.0) * 1000.0


class GazeKalmanFilter:
    def __init__(self, process_noise: float = 1e-3, measurement_noise: float = 5e-2):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float32,
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False

    def reset(self):
        self.initialized = False

    def update(self, measurement: np.ndarray) -> np.ndarray:
        measurement = np.asarray(measurement, dtype=np.float32).reshape(2, 1)

        if not self.initialized:
            self.kf.statePost = np.array(
                [[measurement[0, 0]], [measurement[1, 0]], [0.0], [0.0]],
                dtype=np.float32,
            )
            self.initialized = True
            return measurement[:, 0]

        self.kf.predict()
        estimated = self.kf.correct(measurement)
        return estimated[:2, 0]


class FaceBoxKalmanFilter:
    def __init__(self, process_noise: float = 1e-2, measurement_noise: float = 5e-1):
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)
        self.initialized = False

    def reset(self):
        self.initialized = False

    def update(self, measurement) -> tuple[int, int, int, int]:
        measurement = np.asarray(measurement, dtype=np.float32).reshape(4, 1)

        if not self.initialized:
            self.kf.statePost = np.array(
                [
                    [measurement[0, 0]],
                    [measurement[1, 0]],
                    [measurement[2, 0]],
                    [measurement[3, 0]],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                ],
                dtype=np.float32,
            )
            self.initialized = True
            estimated = measurement[:, 0]
        else:
            self.kf.predict()
            estimated = self.kf.correct(measurement)[:4, 0]

        x, y, w, h = estimated
        return int(round(x)), int(round(y)), int(round(w)), int(round(h))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Real-time webcam gaze demo using the exported ONNX model."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to the ONNX model.")
    parser.add_argument(
        "--camera-id",
        type=int,
        default=1,
        help="Preferred OpenCV camera id. The demo will fall back to other local camera ids if this one has no valid frames.",
    )
    parser.add_argument("--width", type=int, default=640, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=480, help="Requested capture height.")
    parser.add_argument("--face-size", type=int, default=224, help="Model input size for face crops.")
    parser.add_argument(
        "--detect-every",
        type=int,
        default=3,
        help="Run face detection every N frames and reuse the last face box in between.",
    )
    parser.add_argument(
        "--infer-every",
        type=int,
        default=2,
        help="Run gaze inference every N frames and reuse the last gaze in between.",
    )
    parser.add_argument(
        "--detect-scale",
        type=float,
        default=0.6,
        help="Downscale ratio used before face detection. Smaller is faster but may miss small faces.",
    )
    parser.add_argument(
        "--intra-op-threads",
        type=int,
        default=2,
        help="ONNX Runtime intra-op threads. Defaults to 2 based on current local benchmark.",
    )
    parser.add_argument(
        "--inter-op-threads",
        type=int,
        default=0,
        help="ONNX Runtime inter-op threads. Use 0 for runtime default.",
    )
    parser.add_argument(
        "--show-profiler",
        action="store_true",
        help="Show per-stage timing overlay for read, detect, infer, and draw.",
    )
    parser.add_argument(
        "--score-thickness",
        type=int,
        default=2,
        help="Line thickness for the gaze arrow.",
    )
    parser.add_argument(
        "--process-noise",
        type=float,
        default=1e-3,
        help="Kalman filter process noise. Larger values respond faster but smooth less.",
    )
    parser.add_argument(
        "--measurement-noise",
        type=float,
        default=5e-2,
        help="Kalman filter measurement noise. Larger values smooth more but lag more.",
    )
    parser.add_argument(
        "--box-process-noise",
        type=float,
        default=1e-2,
        help="Face box Kalman process noise. Larger values follow movement faster.",
    )
    parser.add_argument(
        "--box-measurement-noise",
        type=float,
        default=5e-1,
        help="Face box Kalman measurement noise. Larger values produce a steadier box.",
    )
    return parser


def create_session(model_path: Path, intra_op_threads: int, inter_op_threads: int) -> ort.InferenceSession:
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
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


def has_valid_frame(cap: cv2.VideoCapture, warmup_frames: int = 20) -> bool:
    for _ in range(warmup_frames):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return True
    return False


def open_camera(preferred_camera_id: int, width: int, height: int) -> cv2.VideoCapture:
    candidate_ids = [preferred_camera_id]
    for candidate in [0, 1, 2, 3, 4]:
        if candidate not in candidate_ids:
            candidate_ids.append(candidate)

    # On Windows, DSHOW and MSMF usually behave better with built-in webcams
    # than CAP_ANY when virtual cameras have been connected recently.
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

    for backend in backends:
        for camera_id in candidate_ids:
            cap = cv2.VideoCapture(camera_id, backend)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            if has_valid_frame(cap):
                print(f"Using camera id {camera_id} with backend {backend}")
                return cap

            cap.release()

    raise RuntimeError(
        f"Failed to open a local camera with valid frames. Tried camera ids {candidate_ids} using DSHOW, MSMF, and ANY."
    )


def preprocess_face(face_bgr: np.ndarray, face_size: int) -> np.ndarray:
    face = cv2.resize(face_bgr, (face_size, face_size), interpolation=cv2.INTER_LINEAR)
    face = face.astype(np.float32) / 255.0
    face = np.transpose(face, (2, 0, 1))
    face = np.expand_dims(face, axis=0)
    return face


def gaze_to_3d(gaze_2d: np.ndarray) -> np.ndarray:
    yaw, pitch = float(gaze_2d[0]), float(gaze_2d[1])
    gaze_3d = np.zeros(3, dtype=np.float32)
    gaze_3d[0] = -np.cos(pitch) * np.sin(yaw)
    gaze_3d[1] = -np.sin(pitch)
    gaze_3d[2] = -np.cos(pitch) * np.cos(yaw)
    return gaze_3d


def draw_gaze(frame: np.ndarray, face_box, gaze_2d: np.ndarray, thickness: int):
    x, y, w, h = face_box
    center = (int(x + w / 2), int(y + h / 2))
    gaze_3d = gaze_to_3d(gaze_2d)
    scale = max(w, h) * 0.9
    end = (
        int(center[0] + gaze_3d[0] * scale),
        int(center[1] + gaze_3d[1] * scale),
    )

    cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 220, 120), 2)
    cv2.arrowedLine(frame, center, end, (0, 0, 255), thickness, tipLength=0.18)
    cv2.circle(frame, center, 3, (255, 255, 0), -1)

    yaw_deg = np.degrees(float(gaze_2d[0]))
    pitch_deg = np.degrees(float(gaze_2d[1]))
    cv2.putText(
        frame,
        f"yaw {yaw_deg:+.1f} deg",
        (x, max(20, y - 28)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"pitch {pitch_deg:+.1f} deg",
        (x, max(40, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def pick_largest_face(faces):
    if len(faces) == 0:
        return None
    return max(faces, key=lambda box: box[2] * box[3])


def clamp_face_box(face_box, frame_shape):
    frame_h, frame_w = frame_shape[:2]
    x, y, w, h = face_box
    w = max(1, min(w, frame_w))
    h = max(1, min(h, frame_h))
    x = max(0, min(x, frame_w - w))
    y = max(0, min(y, frame_h - h))
    return x, y, w, h


def detect_face(cascade, gray_frame: np.ndarray, detect_scale: float):
    if detect_scale <= 0 or detect_scale > 1:
        detect_scale = 1.0

    if detect_scale < 0.999:
        small = cv2.resize(gray_frame, None, fx=detect_scale, fy=detect_scale, interpolation=cv2.INTER_LINEAR)
    else:
        small = gray_frame

    faces = cascade.detectMultiScale(
        small,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(max(24, int(80 * detect_scale)), max(24, int(80 * detect_scale))),
    )
    face = pick_largest_face(faces)
    if face is None:
        return None

    x, y, w, h = [int(v) for v in face]
    if detect_scale < 0.999:
        inv = 1.0 / detect_scale
        x = int(round(x * inv))
        y = int(round(y * inv))
        w = int(round(w * inv))
        h = int(round(h * inv))

    return x, y, w, h


def main():
    args = build_parser().parse_args()

    cascade = cv2.CascadeClassifier(str(DEFAULT_CASCADE))
    if cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade: {DEFAULT_CASCADE}")

    session = create_session(args.model, args.intra_op_threads, args.inter_op_threads)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    gaze_filter = GazeKalmanFilter(
        process_noise=args.process_noise,
        measurement_noise=args.measurement_noise,
    )
    box_filter = FaceBoxKalmanFilter(
        process_noise=args.box_process_noise,
        measurement_noise=args.box_measurement_noise,
    )

    cap = open_camera(args.camera_id, args.width, args.height)

    fps = 0.0
    prev_time = time.perf_counter()
    profiler = PerfAverager()
    frame_index = 0
    last_face = None
    last_gaze = None

    try:
        while True:
            read_start = time.perf_counter()
            ok, frame = cap.read()
            profiler.update("read", time.perf_counter() - read_start)
            if not ok:
                cv2.putText(
                    np.zeros((args.height, args.width, 3), dtype=np.uint8),
                    "Failed to read frame from camera.",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                print("Failed to read frame from camera.")
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detect_start = time.perf_counter()
            should_detect = frame_index % max(1, args.detect_every) == 0 or last_face is None
            if should_detect:
                last_face = detect_face(cascade, gray, args.detect_scale)
            profiler.update("detect", time.perf_counter() - detect_start)
            face = last_face

            if face is not None:
                x, y, w, h = [int(v) for v in face]
                pad_x = int(w * 0.15)
                pad_y = int(h * 0.15)
                x0 = max(0, x - pad_x)
                y0 = max(0, y - pad_y)
                x1 = min(frame.shape[1], x + w + pad_x)
                y1 = min(frame.shape[0], y + h + pad_y)

                face_crop = frame[y0:y1, x0:x1]
                if face_crop.size > 0:
                    infer_start = time.perf_counter()
                    should_infer = frame_index % max(1, args.infer_every) == 0 or last_gaze is None
                    if should_infer:
                        model_input = preprocess_face(face_crop, args.face_size)
                        last_gaze = session.run([output_name], {input_name: model_input})[0][0]
                    profiler.update("infer", time.perf_counter() - infer_start)

                    gaze_smoothed = gaze_filter.update(last_gaze)
                    face_smoothed = box_filter.update((x, y, w, h))
                    face_smoothed = clamp_face_box(face_smoothed, frame.shape)

                    draw_start = time.perf_counter()
                    draw_gaze(frame, face_smoothed, gaze_smoothed, args.score_thickness)
                    profiler.update("draw", time.perf_counter() - draw_start)
            else:
                gaze_filter.reset()
                box_filter.reset()
                last_gaze = None
                cv2.putText(
                    frame,
                    "No face detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )

            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

            cv2.putText(
                frame,
                f"demo fps: {fps:.2f}",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (50, 255, 50),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"detect/{args.detect_every} infer/{args.infer_every} threads/{args.intra_op_threads}",
                (20, frame.shape[0] - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                2,
                cv2.LINE_AA,
            )
            if args.show_profiler:
                cv2.putText(
                    frame,
                    (
                        f"read {profiler.get_ms('read'):.1f}ms  "
                        f"detect {profiler.get_ms('detect'):.1f}ms  "
                        f"infer {profiler.get_ms('infer'):.1f}ms  "
                        f"draw {profiler.get_ms('draw'):.1f}ms"
                    ),
                    (20, frame.shape[0] - 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (180, 255, 180),
                    1,
                    cv2.LINE_AA,
                )
            cv2.putText(
                frame,
                "Press q or ESC to quit",
                (20, frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("HFGAD Webcam Gaze Demo", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            frame_index += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
