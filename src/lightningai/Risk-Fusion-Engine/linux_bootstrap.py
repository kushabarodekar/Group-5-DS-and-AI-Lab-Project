"""
Bootstrap native GL/GLES libraries for MediaPipe on headless Linux hosts.

Lightning.ai blocks `conda create` (single env only) and `conda install` into
cloudspace often fails. We ship pre-vendored libGLESv2 dispatch libraries under
native_libs/linux-x86_64/ and can download more from conda-forge without conda.
"""

from __future__ import annotations

import glob
import io
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
from urllib.request import urlopen

logger = logging.getLogger("DriverWellnessAI.bootstrap")

_GLES_BOOTSTRAPPED = False
_GLES_LIB_PATH: Optional[str] = None
_DOWNLOAD_ATTEMPTED = False
_CONDA_INSTALL_ATTEMPTED = False

_GLES_SO_NAMES = ("libGLESv2.so.2", "libGLESv2.so")
_EGL_SO_NAMES = ("libEGL.so.1", "libEGL.so")
_GLVND_SO_NAMES = ("libGLdispatch.so.0", "libGLdispatch.so")

PROJECT_ROOT = Path(__file__).resolve().parent
NATIVE_GLES_LIB_DIR = PROJECT_ROOT / "native_libs" / "linux-x86_64"

# Bundled / downloadable libglvnd artifacts (linux-64, conda-forge 1.7.0 build 3)
_CONDA_FORGE_LINUX64 = "https://conda.anaconda.org/conda-forge/linux-64"
_VENDORED_CONDA_URLS = (
    f"{_CONDA_FORGE_LINUX64}/libgles-1.7.0-ha4b6fd6_3.conda",
    f"{_CONDA_FORGE_LINUX64}/libglvnd-1.7.0-ha4b6fd6_3.conda",
    f"{_CONDA_FORGE_LINUX64}/libegl-1.7.0-ha4b6fd6_3.conda",
)
_SONAME_LINKS = {
    "libGLESv2.so.2": "libGLESv2.so.2.1.0",
    "libGLESv1_CM.so.1": "libGLESv1_CM.so.1.2.0",
    "libGLdispatch.so.0": "libGLdispatch.so.0.0.0",
    "libEGL.so.1": "libEGL.so.1.1.0",
}

MANUAL_GLES_INSTALL_CMD = (
    "python scripts/provision_gles.py   # downloads libs into native_libs/linux-x86_64"
)


def _prepend_ld_library_path(directory: str) -> None:
    if not directory or not os.path.isdir(directory):
        return
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    if directory not in parts:
        os.environ["LD_LIBRARY_PATH"] = directory + (os.pathsep + existing if existing else "")


def _library_search_dirs() -> List[str]:
    dirs: List[str] = []

    if NATIVE_GLES_LIB_DIR.is_dir():
        dirs.append(str(NATIVE_GLES_LIB_DIR))

    for prefix in (
        os.environ.get("CONDA_PREFIX"),
        os.environ.get("VIRTUAL_ENV"),
        sys.prefix,
    ):
        if prefix:
            lib_dir = os.path.join(prefix, "lib")
            if os.path.isdir(lib_dir):
                dirs.append(lib_dir)

    dirs.extend(
        [
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib/x86_64-linux-gnu/mesa-egl",
            "/usr/lib64",
            "/usr/lib",
            "/lib/x86_64-linux-gnu",
        ]
    )

    for pattern in ("/usr/lib/**/libGLESv2.so*", "/lib/**/libGLESv2.so*"):
        for path in glob.glob(pattern, recursive=True):
            parent = os.path.dirname(path)
            if parent not in dirs:
                dirs.append(parent)

    seen = set()
    unique: List[str] = []
    for directory in dirs:
        if directory not in seen:
            seen.add(directory)
            unique.append(directory)
    return unique


def _find_shared_library(names: Iterable[str]) -> Optional[str]:
    import ctypes.util

    for name in names:
        base = name.replace(".so.2", "").replace(".so.1", "").replace(".so", "")
        found = ctypes.util.find_library(base)
        if found and os.path.isfile(found):
            return found

    for directory in _library_search_dirs():
        for name in names:
            candidate = os.path.join(directory, name)
            if os.path.isfile(candidate):
                return candidate
        for name in names:
            base = name.split(".so")[0]
            for candidate in sorted(glob.glob(os.path.join(directory, f"{base}.so*"))):
                if os.path.isfile(candidate) or os.path.islink(candidate):
                    return candidate
    return None


def _preload_shared_library(path: str) -> bool:
    import ctypes

    try:
        ctypes.CDLL(os.path.realpath(path), mode=ctypes.RTLD_GLOBAL)
        _prepend_ld_library_path(os.path.dirname(os.path.realpath(path)))
        return True
    except OSError as exc:
        logger.debug("Could not preload %s: %s", path, exc)
        return False


def _ensure_soname_links(lib_dir: Path) -> None:
    for link_name, target_name in _SONAME_LINKS.items():
        target = lib_dir / target_name
        link = lib_dir / link_name
        if target.is_file() and not link.exists():
            try:
                os.symlink(target_name, link)
            except OSError:
                pass


def _decompress_zst(data: bytes) -> bytes:
    try:
        import zstandard as zstd

        return zstd.ZstdDecompressor().decompress(data)
    except Exception:
        pass

    zstd_bin = shutil.which("zstd")
    if zstd_bin:
        proc = subprocess.run(
            [zstd_bin, "-d", "--stdout"],
            input=data,
            capture_output=True,
            timeout=120,
        )
        if proc.returncode == 0:
            return proc.stdout
    raise RuntimeError("Need zstandard (pip install zstandard) or zstd CLI to extract conda packages")


def _extract_conda_package(url: str, lib_dir: Path) -> None:
    lib_dir.mkdir(parents=True, exist_ok=True)
    payload = urlopen(url, timeout=120).read()
    archive = zipfile.ZipFile(io.BytesIO(payload))
    keep = ("libGLES", "libGLdispatch", "libEGL")
    for member in archive.namelist():
        if not member.startswith("pkg-") or not member.endswith(".tar.zst"):
            continue
        tar_bytes = _decompress_zst(archive.read(member))
        with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tf:
            for entry in tf.getmembers():
                if not entry.isfile():
                    continue
                base = Path(entry.name).name
                if not any(token in base for token in keep):
                    continue
                extracted = tf.extractfile(entry)
                if extracted is None:
                    continue
                (lib_dir / base).write_bytes(extracted.read())
                logger.info("Extracted %s", base)
    _ensure_soname_links(lib_dir)


def download_vendored_gles_libs(force: bool = False) -> Optional[Path]:
    """Download libGLESv2 dispatch libs from conda-forge into native_libs/ (no conda CLI)."""
    global _DOWNLOAD_ATTEMPTED
    if _DOWNLOAD_ATTEMPTED and not force:
        if (NATIVE_GLES_LIB_DIR / "libGLESv2.so.2").exists():
            return NATIVE_GLES_LIB_DIR
        return None
    _DOWNLOAD_ATTEMPTED = True

    if (NATIVE_GLES_LIB_DIR / "libGLESv2.so.2").exists() and not force:
        return NATIVE_GLES_LIB_DIR

    try:
        for url in _VENDORED_CONDA_URLS:
            _extract_conda_package(url, NATIVE_GLES_LIB_DIR)
        _ensure_soname_links(NATIVE_GLES_LIB_DIR)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not download vendored GL libs: %s", exc)
        return None

    if (NATIVE_GLES_LIB_DIR / "libGLESv2.so.2").exists():
        logger.info("Vendored libGLESv2 at %s", NATIVE_GLES_LIB_DIR / "libGLESv2.so.2")
        return NATIVE_GLES_LIB_DIR
    return None


def _run_conda_install_existing_env(packages: Sequence[str], no_deps: bool = False) -> bool:
    installer = shutil.which("mamba") or shutil.which("conda")
    if not installer:
        return False
    cmd = [
        installer,
        "install",
        "-y",
        "--override-channels",
        "-c",
        "conda-forge",
    ]
    if no_deps:
        cmd.append("--no-deps")
    cmd.extend(packages)
    logger.info("Trying install into active env: %s", " ".join(packages))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if result.returncode == 0:
            return True
        combined = (result.stderr or "") + (result.stdout or "")
        if "Conda create is not allowed" in combined:
            return False
        logger.warning("conda install failed: %s", combined[:800])
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("conda install failed: %s", exc)
        return False


def try_conda_install_gles() -> bool:
    """Install into the single allowed cloudspace env (no conda create)."""
    global _CONDA_INSTALL_ATTEMPTED
    if _CONDA_INSTALL_ATTEMPTED:
        return False
    _CONDA_INSTALL_ATTEMPTED = True

    for package_set in (
        ("libgles", "libglvnd", "libegl"),
        ("libgles", "libglvnd"),
    ):
        if _run_conda_install_existing_env(package_set, no_deps=True):
            return True
        if _run_conda_install_existing_env(package_set, no_deps=False):
            return True
    return False


def provision_gles_libraries(force: bool = False) -> Optional[Path]:
    """Best-effort GL provisioning for Lightning.ai (no conda create)."""
    if (NATIVE_GLES_LIB_DIR / "libGLESv2.so.2").exists() and not force:
        return NATIVE_GLES_LIB_DIR

    downloaded = download_vendored_gles_libs(force=force)
    if downloaded is not None:
        return downloaded

    if try_conda_install_gles():
        conda_lib = Path(os.environ.get("CONDA_PREFIX", "")) / "lib"
        if (conda_lib / "libGLESv2.so.2").exists():
            return conda_lib

    return None


def ensure_gles_preloaded(auto_install: bool = True, force_rescan: bool = False) -> Optional[str]:
    global _GLES_BOOTSTRAPPED, _GLES_LIB_PATH
    if _GLES_BOOTSTRAPPED and not force_rescan:
        return _GLES_LIB_PATH
    if force_rescan:
        _GLES_LIB_PATH = None
    _GLES_BOOTSTRAPPED = True

    if sys.platform != "linux":
        return None

    os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

    for directory in _library_search_dirs():
        _prepend_ld_library_path(directory)

    def _attempt_load() -> Optional[str]:
        for names in (_GLVND_SO_NAMES, _EGL_SO_NAMES):
            path = _find_shared_library(names)
            if path:
                _preload_shared_library(path)
        gles_path = _find_shared_library(_GLES_SO_NAMES)
        if gles_path and _preload_shared_library(gles_path):
            logger.info("Preloaded MediaPipe GL dependency: %s", gles_path)
            return gles_path
        return None

    loaded = _attempt_load()
    if loaded:
        _GLES_LIB_PATH = loaded
        return loaded

    if auto_install:
        if provision_gles_libraries():
            for directory in _library_search_dirs():
                _prepend_ld_library_path(directory)
            loaded = _attempt_load()
            if loaded:
                _GLES_LIB_PATH = loaded
                return loaded

    logger.warning(
        "libGLESv2 not found. Run: python scripts/provision_gles.py\n"
        "Then: export LD_LIBRARY_PATH=\"${PWD}/native_libs/linux-x86_64:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}\""
    )
    return None


def gles_install_hint() -> str:
    return (
        "MediaPipe could not load libGLESv2 on this Linux host.\n"
        "Lightning.ai blocks conda create — use bundled/downloaded libs instead:\n"
        "  python scripts/provision_gles.py\n"
        "  export LD_LIBRARY_PATH=\"${PWD}/native_libs/linux-x86_64:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}\"\n"
        "Then restart: python app.py"
    )


def diagnose_gles() -> str:
    lines = [
        f"CONDA_PREFIX={os.environ.get('CONDA_PREFIX', '')}",
        f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}",
        f"Bundled lib dir={NATIVE_GLES_LIB_DIR}",
        f"Bundled libGLESv2 exists={(NATIVE_GLES_LIB_DIR / 'libGLESv2.so.2').exists()}",
    ]
    hits = []
    for directory in _library_search_dirs():
        for name in _GLES_SO_NAMES:
            candidate = os.path.join(directory, name)
            if os.path.exists(candidate):
                hits.append(candidate)
    lines.append("libGLESv2 hits: " + (", ".join(hits) if hits else "NONE"))
    return "\n".join(lines)
