#!/usr/bin/env python3
"""
Kaggle GPU fallback launcher — Milestone 6 live demo.

Runs the same integrated Gradio app as app.py (wellness_core + report_dashboard)
but tuned for Kaggle notebook sessions:
  - GPU enabled (Settings → Accelerator → GPU)
  - share=True  → temporary public *.gradio.live URL for evaluators
  - server_name=0.0.0.0 for Kaggle proxy

Does NOT duplicate inference logic — imports app.launch_app() directly.

Usage (Kaggle notebook cell or terminal):
    !python run_kaggle_gradio_fallback.py

Local smoke test (CPU/GPU):
    python run_kaggle_gradio_fallback.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _is_kaggle() -> bool:
    return os.path.isdir("/kaggle/input") or os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None


def _bootstrap_linux_gles() -> None:
    """MediaPipe landmark module needs libGLESv2 on headless Linux (Kaggle included)."""
    if sys.platform != "linux":
        return
    os.environ.setdefault("DW_AUTO_INSTALL_GLES", "1")
    gles_dir = ROOT / "native_libs" / "linux-x86_64"
    if gles_dir.is_dir():
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        gles_path = str(gles_dir)
        if gles_path not in existing.split(os.pathsep):
            os.environ["LD_LIBRARY_PATH"] = (
                gles_path + (os.pathsep + existing if existing else "")
            )


def _print_environment() -> None:
    print("=" * 72)
    print("Risk Fusion Engine — Kaggle GPU Fallback (M6)")
    print("=" * 72)
    print(f"Project root : {ROOT}")
    print(f"Kaggle env   : {_is_kaggle()}")
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        print(f"CUDA         : {cuda_ok}")
        if cuda_ok:
            print(f"GPU device   : {torch.cuda.get_device_name(0)}")
    except Exception as exc:  # noqa: BLE001
        print(f"CUDA check skipped: {exc}")
    print("-" * 72)
    print("Launching integrated Gradio pipeline (same as app.py)...")
    print("  share=True  → public Gradio URL will appear below when ready")
    print("  Keep this notebook/cell running for the duration of the demo.")
    print("=" * 72)


def main() -> None:
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    _bootstrap_linux_gles()
    _print_environment()

    # Import after bootstrap so wellness_core sees LD_LIBRARY_PATH on Linux.
    import app  # noqa: WPS433 — loads models via wellness_core.build_manager()

    app.launch_app(
        share=True,
        debug=True,
        server_name="0.0.0.0",
        server_port=7860,
    )


if __name__ == "__main__":
    main()
