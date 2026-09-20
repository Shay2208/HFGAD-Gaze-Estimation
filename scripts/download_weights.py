#!/usr/bin/env python3
"""Download HFGAD model weights from GitHub Releases.

Why this script?
    Model weights are ~580 MB in total, so they are NOT stored in git.
    They are published as assets of a GitHub Release instead.

Usage
-----
    # Core weights only (Gaze360 .pt + .onnx): enough for the webcam demo
    python scripts/download_weights.py

    # Everything, including the MPIIFaceGaze folds and the ResNet-18 backbone
    python scripts/download_weights.py --all

    # Pin a specific release tag
    python scripts/download_weights.py --tag v1.0.0

    # Use an explicit repository (default: read from `git remote get-url origin`)
    python scripts/download_weights.py --repo <OWNER>/<REPO>

Notes
-----
* The repository is auto-detected from the `origin` remote, so you normally do
  not need `--repo`.
* Already-downloaded files are skipped. Pass `--force` to re-download.
* For private repositories / rate limits, set a token:
      export GITHUB_TOKEN=ghp_xxx
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

API_ROOT = "https://api.github.com"

# Release asset name  ->  destination path relative to the repository root.
ASSETS: dict[str, str] = {
    # ---- Gaze360 (10.46 deg), ResNet-18 backbone --------------------------------
    "hfgad_gaze360_resnet18.pt": "evaluation/gaze360/checkpoint/ours_10.46.pt",
    "hfgad_gaze360_resnet18.onnx": "evaluation/gaze360/checkpoint/ours_10.46.onnx",
    # ---- MPIIFaceGaze, leave-one-out folds ---------------------------------------
    "hfgad_mpii_p00.pt": "evaluation/mpii/checkpoint/p00.label/hfgad.pt",
    "hfgad_mpii_p00.onnx": "evaluation/mpii/checkpoint/p00.label/hfgad.onnx",
    "hfgad_mpii_p04.pt": "evaluation/mpii/checkpoint/p04.label/hfgad.pt",
    "hfgad_mpii_p05.pt": "evaluation/mpii/checkpoint/p05.label/hfgad.pt",
    # ---- Backbones (optional, for training from scratch) -------------------------
    "resnet18-5c106cde.pth": "ckpts/resnet18-5c106cde.pth",
}

# Downloaded by default: the smallest set that makes `webcam_gaze_demo.py` work.
DEFAULT_ASSETS = [
    "hfgad_gaze360_resnet18.pt",
    "hfgad_gaze360_resnet18.onnx",
]


def _slug_to_repo(slug: str) -> str | None:
    """Turn any GitHub remote URL into an `owner/repo` slug."""
    slug = slug.strip()
    if slug.endswith(".git"):
        slug = slug[: -len(".git")]
    patterns = [
        r"^git@github\.com:(?P<owner>[^/]+)/(?P<repo>[^/]+)$",
        r"^ssh://git@github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)$",
        r"^https?://(?:www\.)?github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)$",
        r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)\.git$",
    ]
    for pattern in patterns:
        match = re.match(pattern, slug)
        if match:
            return f"{match.group('owner')}/{match.group('repo')}"
    if re.match(r"^[^/]+/[^/]+$", slug):  # already `owner/repo`
        return slug
    return None


def detect_repo() -> str | None:
    try:
        url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=ROOT_DIR,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", "ignore")
    except Exception:
        return None
    return _slug_to_repo(url)


def _request(url: str) -> bytes:
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "hfgad-weight-downloader")
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def fetch_assets(repo: str, tag: str | None) -> dict[str, str]:
    """Return {asset_name: browser_download_url} for the chosen release."""
    if tag:
        url = f"{API_ROOT}/repos/{repo}/releases/tags/{tag}"
    else:
        url = f"{API_ROOT}/repos/{repo}/releases/latest"
    try:
        payload = json.loads(_request(url))
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            f"Could not fetch release metadata from {url}\n"
            f"  HTTP {exc.code}: {exc.reason}\n"
            "  -> Did you create a Release and attach the weight files?\n"
            "  -> Or pass --repo <OWNER>/<REPO> / --tag <TAG> explicitly."
        ) from exc
    return {a["name"]: a["browser_download_url"] for a in payload.get("assets", [])}


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "hfgad-weight-downloader")
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length", 0) or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100.0 / total
                print(f"\r    {pct:6.2f}%  {done/1e6:7.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download HFGAD model weights from GitHub Releases."
    )
    parser.add_argument("--repo", default=None, help="GitHub repository as OWNER/REPO.")
    parser.add_argument("--tag", default=None, help="Release tag (default: latest).")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download every known asset, not just the demo weights.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download existing files.")
    parser.add_argument("--list", action="store_true", help="List release assets and exit.")
    args = parser.parse_args()

    repo = args.repo or detect_repo()
    if not repo:
        raise SystemExit(
            "Cannot detect the GitHub repository.\n"
            "Add a remote first:  git remote add origin https://github.com/<OWNER>/<REPO>.git\n"
            "or pass --repo <OWNER>/<REPO> explicitly."
        )
    print(f"Repository: {repo}")

    available = fetch_assets(repo, args.tag)

    if args.list:
        print("Assets in this release:")
        for name in sorted(available):
            print(f"  - {name}")
        return 0

    wanted = list(ASSETS.keys()) if args.all else DEFAULT_ASSETS

    ok, missing = 0, []
    for name in wanted:
        dest = ROOT_DIR / ASSETS[name]
        if dest.exists() and not args.force:
            print(f"[skip] {name} -> {dest.relative_to(ROOT_DIR)} (exists)")
            ok += 1
            continue
        if name not in available:
            missing.append(name)
            continue
        print(f"[get ] {name} -> {dest.relative_to(ROOT_DIR)}")
        download(available[name], dest)
        ok += 1

    if missing:
        print("\nNot found in the release (skipped):")
        for name in missing:
            print(f"  - {name}")
        print("\nAttach them to the Release, then re-run this script.")

    print(f"\nDone. {ok} file(s) ready.")
    if ok:
        print("Next:  python webcam_gaze_demo.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
