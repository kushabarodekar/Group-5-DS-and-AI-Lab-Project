#!/usr/bin/env bash
# Linux GL/GLES check for MediaPipe (Lightning.ai safe — no conda create).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

have_cmd() { command -v "$1" >/dev/null 2>&1; }

echo "=== Driver Wellness AI — Linux GL dependency check ==="
echo "Project: $ROOT_DIR"
echo ""

BUNDLED="$ROOT_DIR/native_libs/linux-x86_64/libGLESv2.so.2"
if [ -f "$BUNDLED" ] || [ -L "$BUNDLED" ]; then
  echo "  libGLESv2 (bundled): $BUNDLED"
  GLES_OK=1
else
  echo "  libGLESv2 (bundled): NOT FOUND — running provision_gles.py"
  GLES_OK=0
  python3 scripts/provision_gles.py || true
  if [ -f "$BUNDLED" ] || [ -L "$BUNDLED" ]; then
    GLES_OK=1
    echo "  libGLESv2 (downloaded): $BUNDLED"
  fi
fi

if have_cmd ffmpeg; then
  echo "  ffmpeg: $(ffmpeg -version | head -n 1)"
else
  echo "  ffmpeg: NOT FOUND (optional — needed for browser video re-encode)"
fi

echo ""
if [ "$GLES_OK" -eq 1 ]; then
  export LD_LIBRARY_PATH="$ROOT_DIR/native_libs/linux-x86_64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  echo "OK. Start the app with:"
  echo "  export LD_LIBRARY_PATH=\"\${PWD}/native_libs/linux-x86_64:\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}\""
  echo "  python app.py"
  exit 0
fi

echo "FAILED: libGLESv2 still missing."
echo "  pip install zstandard"
echo "  python scripts/provision_gles.py"
exit 1
