#!/usr/bin/env python3
"""Download or verify vendored libGLESv2 libs (no conda create — Lightning.ai safe)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import linux_bootstrap


def main() -> int:
    print("=== MediaPipe GL / GLES provisioning (Lightning.ai safe) ===")
    print(linux_bootstrap.diagnose_gles())
    print()

    bundled = linux_bootstrap.NATIVE_GLES_LIB_DIR / "libGLESv2.so.2"
    if bundled.exists():
        print(f"Bundled libs already present: {bundled}")
    else:
        print("Downloading libGLESv2 dispatch libraries from conda-forge (no conda CLI)...")
        lib_dir = linux_bootstrap.download_vendored_gles_libs(force=True)
        if lib_dir is None:
            print("Download failed.")
            print("Install extractor: pip install zstandard   (or ensure `zstd` is on PATH)")
            print("Then re-run: python scripts/provision_gles.py")
            return 1

    gles = linux_bootstrap.NATIVE_GLES_LIB_DIR / "libGLESv2.so.2"
    if not gles.exists():
        print(f"FAILED: {gles} still missing")
        return 1

    print(f"SUCCESS: {gles}")
    print()
    print("Before starting the app:")
    print('  export LD_LIBRARY_PATH="${PWD}/native_libs/linux-x86_64:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"')
    print("  python app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
