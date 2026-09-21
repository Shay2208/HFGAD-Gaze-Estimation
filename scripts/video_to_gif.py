#!/usr/bin/env python
"""Convert a recorded demo video into a README-ready, size-optimised GIF.

The webcam demo writes an MP4 with ``--record-output``. GitHub renders GIFs
inline in READMEs but not MP4s, and a raw MP4 is far too heavy to commit, so
this script turns the recording into a small looping GIF.

It prefers FFmpeg (either from PATH or the ``imageio-ffmpeg`` wheel) because its
two-pass palette pipeline produces a much better looking GIF than naive frame
quantisation. If no FFmpeg is available it falls back to imageio/Pillow.

Usage
-----
    python scripts/video_to_gif.py assets/demo.mp4
    python scripts/video_to_gif.py assets/demo.mp4 -o assets/demo.gif --fps 12 --width 720
    python scripts/video_to_gif.py assets/demo.mp4 --start 2 --duration 8 --speed 1.2

Then uncomment the demo block near the top of README.md.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT_DIR / "assets" / "demo.gif"

# GitHub renders inline images fine, but keeps the page light. Around 6-10 MB is
# the practical ceiling for a README GIF; above that reviewers on slow links
# simply never see it.
SIZE_WARN_BYTES = 8 * 1024 * 1024


def resolve_ffmpeg() -> str | None:
    """Find an ffmpeg executable: system PATH first, then the imageio wheel."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg  # type: ignore

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return str(exe)
    except Exception:
        pass
    return None


def build_filter_chain(fps: float, width: int | None, speed: float) -> str:
    parts = []
    if speed and abs(speed - 1.0) > 1e-6:
        parts.append(f"setpts=PTS/{speed:.4f}")
    if width and width > 0:
        parts.append(f"fps={fps:g},scale={width}:-1:flags=lanczos")
    else:
        parts.append(f"fps={fps:g}")
    return ",".join(parts)


def convert_with_ffmpeg(
    ffmpeg: str,
    src: Path,
    dst: Path,
    fps: float,
    width: int | None,
    start: float,
    duration: float | None,
    speed: float,
    colors: int,
    loop: int,
) -> None:
    trim = []
    if start > 0:
        trim += ["-ss", f"{start:g}"]
    if duration and duration > 0:
        trim += ["-t", f"{duration:g}"]

    pre = build_filter_chain(fps, width, speed)
    filter_complex = (
        f"{pre},split[a][b];"
        f"[a]palettegen=max_colors={colors}:stats_mode=diff[p];"
        f"[b][p]paletteuse=dither=bayer:bayer_scale=3"
    )

    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        *trim,
        "-i",
        str(src),
        "-filter_complex",
        filter_complex,
        "-loop",
        str(loop),
        str(dst),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def convert_with_imageio(
    src: Path,
    dst: Path,
    fps: float,
    width: int | None,
    start: float,
    duration: float | None,
    speed: float,
) -> None:
    """Fallback path: decode with OpenCV, encode with imageio/Pillow."""
    import cv2  # noqa: WPS433  (optional dependency, only used in fallback)
    import imageio.v2 as imageio  # noqa: WPS433

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(start * src_fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = None
    if duration and duration > 0:
        max_frames = int(duration * src_fps)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    step = max(1, int(round(src_fps / max(fps, 1e-3))))
    frames = []
    index = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and index >= max_frames:
            break
        if index % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if width and width > 0 and rgb.shape[1] != width:
                height = int(round(rgb.shape[0] * width / rgb.shape[1]))
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
            frames.append(rgb)
            written += 1
        index += 1
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from {src} (total frames reported: {total_frames})")

    duration_ms = (1000.0 / (fps * speed)) if speed else (1000.0 / fps)
    dst.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(dst), frames, format="GIF", duration=duration_ms / 1000.0, loop=0)
    print(f"Wrote {written} frames via imageio fallback.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a recorded demo video (MP4/MOV/WebM) into an optimised looping GIF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Source video recorded by webcam_gaze_demo.py.")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT, help="Output GIF path.")
    parser.add_argument("--fps", type=float, default=12.0, help="GIF frame rate. 10-15 keeps files small.")
    parser.add_argument(
        "--width",
        type=int,
        default=720,
        help="Output width in pixels, height scaled to keep aspect ratio. Use 0 to keep the source size.",
    )
    parser.add_argument("--start", type=float, default=0.0, help="Skip this many seconds from the beginning.")
    parser.add_argument("--duration", type=float, default=0.0, help="Only convert this many seconds. 0 = whole video.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (>1 is faster).")
    parser.add_argument("--colors", type=int, default=128, help="Palette size. Lower means a smaller file.")
    parser.add_argument("--loop", type=int, default=0, help="GIF loop count. 0 = loop forever.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src: Path = args.input
    if not src.exists():
        print(f"Input video not found: {src}")
        return 1

    dst: Path = args.output
    dst.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = resolve_ffmpeg()
    if ffmpeg:
        convert_with_ffmpeg(
            ffmpeg,
            src,
            dst,
            args.fps,
            args.width if args.width > 0 else None,
            args.start,
            args.duration if args.duration > 0 else None,
            args.speed,
            args.colors,
            args.loop,
        )
    else:
        print("ffmpeg not found - falling back to imageio (lower quality, larger files).")
        print("Tip: install a bundled ffmpeg with `pip install imageio-ffmpeg`.")
        convert_with_imageio(
            src,
            dst,
            args.fps,
            args.width if args.width > 0 else None,
            args.start,
            args.duration if args.duration > 0 else None,
            args.speed,
        )

    size = dst.stat().st_size
    print(f"GIF written: {dst} ({size / 1024 / 1024:.2f} MB)")
    if size > SIZE_WARN_BYTES:
        print(
            "Warning: the GIF is large for a README. Try --fps 10 --width 560 --colors 64, "
            "or trim it with --start/--duration."
        )
    print("Next: uncomment the demo block near the top of README.md and commit assets/demo.gif.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
