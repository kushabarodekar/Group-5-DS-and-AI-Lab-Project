"""
Driver Wellness AI — Hugging Face Spaces entry point (Gradio).

Thin UI layer over wellness_core.py + report_dashboard.py.
"""

from __future__ import annotations

import inspect
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Optional

# MediaPipe needs libGLESv2 on Linux — bootstrap before wellness_core imports mediapipe.
if sys.platform == "linux":
    try:
        import linux_bootstrap

        _auto = __import__("os").environ.get("DW_AUTO_INSTALL_GLES", "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )
        linux_bootstrap.ensure_gles_preloaded(auto_install=_auto)
    except Exception:
        pass

import gradio as gr

try:
    import spaces  # noqa: F401

    def gpu(duration=120):
        return spaces.GPU(duration=duration)
except Exception:  # pragma: no cover
    def gpu(duration=120):
        def _wrap(fn):
            return fn
        return _wrap

import wellness_core as core
import report_dashboard as dashboard

APP_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = APP_DIR / "outputs"

print("Loading models... (first startup can take a minute)")
MANAGER = core.build_manager()
print("Models loaded.")

GRADIO_MAJOR = int(gr.__version__.split(".")[0])


def _supports_param(component_cls, param_name: str) -> bool:
    try:
        return param_name in inspect.signature(component_cls.__init__).parameters
    except (TypeError, ValueError):
        return False


def _blocks_kwargs() -> dict:
    kwargs = {"title": "Driver Wellness AI"}
    if _supports_param(gr.Blocks, "fill_width"):
        kwargs["fill_width"] = False
    if GRADIO_MAJOR < 6:
        kwargs["theme"] = APP_THEME
        kwargs["css"] = CUSTOM_CSS
    return kwargs


def _video_output(label: str) -> gr.Video:
    """Single output video component (browser-playable MP4)."""
    kwargs = {"label": label, "interactive": False}
    if _supports_param(gr.Video, "format"):
        kwargs["format"] = "mp4"
    if _supports_param(gr.Video, "autoplay"):
        kwargs["autoplay"] = True
    return gr.Video(**kwargs)


def _resolve_upload_path(video_input) -> Optional[str]:
    """Normalize Gradio 5/6 video input shapes to a local filepath."""
    if not video_input:
        return None
    if isinstance(video_input, str):
        return video_input
    if isinstance(video_input, Path):
        return str(video_input)
    if isinstance(video_input, dict):
        for key in ("video", "path", "name"):
            val = video_input.get(key)
            if val:
                return str(val)
    return str(video_input)


def _video_output_value(path: Optional[str]) -> Optional[str]:
    """Return absolute filepath for Gradio Video / DownloadButton (str only — not FileData)."""
    if not path:
        return None
    resolved = Path(path).resolve()
    if not resolved.is_file():
        return None
    return str(resolved)


APP_THEME = gr.themes.Soft(
    primary_hue="orange",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

CUSTOM_CSS = """
:root {
  --dw-primary: #e67e22;
  --dw-primary-dark: #d35400;
  --dw-bg: #f4f6f8;
  --dw-card: #ffffff;
  --dw-text: #2c3e50;
  --dw-muted: #7f8c8d;
  --dw-border: #e1e8ed;
  --dw-success: #2ecc71;
  --dw-warning: #f1c40f;
  --dw-danger: #e74c3c;
}
body {
  background: var(--dw-bg) !important;
}
.gradio-container {
  font-family: "Inter", "Segoe UI", system-ui, sans-serif !important;
  background: linear-gradient(180deg, #fafbfc 0%, var(--dw-bg) 120px) !important;
  width: min(1180px, 94vw) !important;
  max-width: 1180px !important;
  margin: 0 auto !important;
  padding: 1rem 1.25rem 2rem !important;
}
main.app, .app {
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
}
.contain {
  width: 100% !important;
  max-width: 1180px !important;
  margin: 0 auto !important;
}
.hero-card {
  background: var(--dw-card);
  border: 1px solid var(--dw-border);
  border-radius: 16px;
  padding: 1.25rem 1.5rem;
  margin-bottom: 1rem;
  box-shadow: 0 8px 24px rgba(44, 62, 80, 0.06);
}
.hero-card h1 { margin: 0 0 0.35rem 0; color: var(--dw-text); font-size: 1.75rem; }
.hero-card p { margin: 0; color: var(--dw-muted); line-height: 1.5; }
.panel-card {
  background: var(--dw-card);
  border: 1px solid var(--dw-border);
  border-radius: 14px;
  padding: 1rem;
  box-shadow: 0 4px 16px rgba(44, 62, 80, 0.04);
}
.status-pill {
  display: inline-block;
  padding: 0.25rem 0.65rem;
  border-radius: 999px;
  font-size: 0.8rem;
  font-weight: 600;
  background: #eef2f7;
  color: var(--dw-text);
}
#analyze-btn {
  background: linear-gradient(135deg, var(--dw-primary), var(--dw-primary-dark)) !important;
  border: none !important;
  font-weight: 600 !important;
  transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
#analyze-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 20px rgba(230, 126, 34, 0.35) !important;
}
.dw-report h2, .dw-report h3 { color: var(--dw-text); margin-top: 1rem; }
.dw-report .score-header { text-align: center; margin: 0.5rem 0 1rem; }
.dw-report .score-value { font-size: 2rem; font-weight: 700; color: var(--dw-text); }
.dw-report .score-value span { font-size: 1rem; color: var(--dw-muted); font-weight: 500; }
.dw-report .risk-badge {
  display: inline-block;
  color: #fff;
  padding: 0.35rem 0.9rem;
  border-radius: 999px;
  font-weight: 600;
  margin-top: 0.5rem;
}
.dw-report .gauge-track {
  position: relative;
  display: flex;
  height: 18px;
  border-radius: 999px;
  overflow: hidden;
  margin: 0.75rem 0 0.35rem;
}
.dw-report .gauge-zone { flex: 1; }
.dw-report .gauge-zone.low { background: var(--dw-success); }
.dw-report .gauge-zone.medium { background: var(--dw-warning); }
.dw-report .gauge-zone.high { background: #e67e22; }
.dw-report .gauge-zone.critical { background: var(--dw-danger); }
.dw-report .gauge-marker {
  position: absolute;
  top: -4px;
  width: 4px;
  height: 26px;
  background: #111;
  transform: translateX(-50%);
  border-radius: 2px;
}
.dw-report .gauge-caption { text-align: center; color: var(--dw-muted); font-size: 0.85rem; }
.dw-report .module-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 0.75rem;
  margin: 0.75rem 0 1rem;
}
.dw-report .module-card {
  border: 2px solid;
  border-radius: 12px;
  padding: 0.75rem;
  min-height: 110px;
}
.dw-report .module-name { font-size: 0.78rem; font-weight: 700; margin-bottom: 0.35rem; }
.dw-report .module-pred { font-size: 0.95rem; font-weight: 600; margin-bottom: 0.25rem; }
.dw-report .module-meta { font-size: 0.78rem; color: var(--dw-muted); }
.dw-report .fusion-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
  margin: 0.5rem 0 1rem;
}
.dw-report .fusion-table th, .dw-report .fusion-table td {
  border: 1px solid var(--dw-border);
  padding: 0.45rem 0.55rem;
  text-align: center;
}
.dw-report .fusion-table th { background: #f8fafc; font-weight: 600; }
.dw-report .session-details {
  background: #f8fafc;
  border: 1px solid var(--dw-border);
  border-radius: 10px;
  padding: 0.65rem 0.85rem;
}
.dw-report .session-details summary { cursor: pointer; font-weight: 600; }
.dw-live-status .live-module-list {
  margin: 0.75rem 0 0;
  padding-left: 1.1rem;
  color: var(--dw-text);
  font-size: 0.9rem;
}
.dw-live-status .live-module-list li { margin: 0.25rem 0; }
.dw-live-status .live-warming {
  color: var(--dw-muted);
  font-size: 0.85rem;
  margin: 0.35rem 0;
}
footer { display: none !important; }
"""


def _empty_recorded_outputs():
    return None, None, "", None, None, "<span class='status-pill'>Upload a video and click Analyze.</span>"


@gpu(duration=120)
def analyze_video(video_input):
    video_path = _resolve_upload_path(video_input)
    if not video_path:
        return _empty_recorded_outputs()
    try:
        payload = core.run_recorded_video(MANAGER, video_path)
        assets = dashboard.build_report_assets(payload["analysis"], payload["summary"])
        video_value = _video_output_value(payload["video_path"])
        status = (
            f"<span class='status-pill' style='background:#e8f8ef;color:#1e8449;'>"
            f"Analysis complete · {payload['summary'].get('Frames processed', '?')} frames</span>"
        )
        return (
            video_value,
            video_value,
            assets["html"],
            assets["dashboard_png"],
            assets["dashboard_pdf"],
            status,
        )
    except Exception:
        err = html_escape(traceback.format_exc())
        return (
            None,
            None,
            f"<pre style='color:#c0392b;white-space:pre-wrap;'>{err}</pre>",
            None,
            None,
            "<span class='status-pill' style='background:#fdecea;color:#c0392b;'>Analysis failed</span>",
        )


def html_escape(text: str) -> str:
    import html as html_mod
    return html_mod.escape(text)


def live_start():
    core.start_live_session()
    return None, dashboard.build_live_status_html({}, 0, 0.0)


def live_stream(frame):
    if frame is None:
        return None, dashboard.build_live_status_html({}, 0, 0.0)
    try:
        annotated_rgb, fusion, frame_count, fps = core.process_live_frame(frame)
        return annotated_rgb, dashboard.build_live_status_html(fusion, frame_count, fps)
    except Exception:
        err = html_escape(traceback.format_exc())
        return frame, f"<pre style='color:#c0392b;'>{err}</pre>"


def live_stop():
    try:
        payload = core.stop_live_session()
        if payload is None:
            return (
                "<p class='status-pill'>No active live session.</p>",
                None,
                None,
            )
        assets = dashboard.build_report_assets(payload["analysis"], payload["summary"])
        return assets["html"], assets["dashboard_png"], assets["dashboard_pdf"]
    except Exception:
        err = html_escape(traceback.format_exc())
        return f"<pre style='color:#c0392b;'>{err}</pre>", None, None


try:
    gr.set_static_paths(paths=[str(OUTPUT_DIR.resolve())])
except Exception:
    pass

with gr.Blocks(**_blocks_kwargs()) as demo:
    gr.HTML(
        """
        <div class="hero-card">
          <h1>Driver Wellness AI</h1>
          <p>
            Five specialized models — video fatigue, landmark fatigue, driver activity,
            seat-belt/phone, and smoking/drinking — fused into one wellness score using
            the Common Driver Risk Score Framework (Option A, exponential).
          </p>
        </div>
        """
    )

    with gr.Tab("Recorded video"):
        gr.Markdown("Upload a short driving clip to get a fused, annotated analysis.")
        status_bar = gr.HTML("<span class='status-pill'>Ready</span>")

        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                with gr.Group(elem_classes=["panel-card"]):
                    inp = gr.Video(label="Driving clip", sources=["upload"], interactive=True, format="mp4")
                    analyze_btn = gr.Button("Analyze", variant="primary", elem_id="analyze-btn")

            with gr.Column(scale=7):
                with gr.Group(elem_classes=["panel-card"]):
                    out_video = _video_output("Annotated output")
                    video_download = gr.DownloadButton(
                        "Download annotated video",
                        value=None,
                        variant="secondary",
                    )
                with gr.Accordion("Dashboard chart", open=True):
                    dashboard_img = gr.Image(label="Colab-style dashboard", type="filepath", interactive=False)
                with gr.Accordion("Detailed report", open=True):
                    report_html = gr.HTML(label="Session report")
                with gr.Row():
                    pdf_download = gr.DownloadButton(
                        "Download Report as PDF",
                        value=None,
                        variant="secondary",
                    )

        analyze_btn.click(
            fn=analyze_video,
            inputs=[inp],
            outputs=[out_video, video_download, report_html, dashboard_img, pdf_download, status_bar],
            show_progress="full",
        )

    with gr.Tab("Live webcam"):
        gr.Markdown(
            "Stream your webcam for real-time fused analysis. Click **Start / Reset**, "
            "allow camera access, and the annotated feed updates continuously. "
            "Click **Finish & summarize** for the full Colab-style report.\n\n"
            "> On CPU-only hosts this runs slowly; GPU hardware improves real-time performance."
        )
        with gr.Row():
            with gr.Column():
                with gr.Group(elem_classes=["panel-card"]):
                    cam = gr.Image(
                        label="Webcam input",
                        sources=["webcam"],
                        streaming=True,
                        type="numpy",
                    )
                    with gr.Row():
                        start_btn = gr.Button("Start / Reset session", variant="primary")
                        stop_btn = gr.Button("Finish & summarize")
            with gr.Column():
                with gr.Group(elem_classes=["panel-card"]):
                    live_out = gr.Image(label="Annotated live feed", type="numpy")
                    live_text = gr.HTML(dashboard.build_live_status_html({}, 0, 0.0))
        with gr.Accordion("Session report", open=True):
            live_report_html = gr.HTML("<p class='status-pill'>Finish the session to generate the full report.</p>")
        with gr.Accordion("Dashboard chart", open=False):
            live_dashboard_img = gr.Image(label="Colab-style dashboard", type="filepath", interactive=False)
        live_pdf_download = gr.DownloadButton(
            "Download Report as PDF",
            value=None,
            variant="secondary",
        )

        start_btn.click(live_start, inputs=None, outputs=[live_out, live_text])
        cam.stream(
            live_stream,
            inputs=[cam],
            outputs=[live_out, live_text],
            stream_every=0.1,
            concurrency_limit=1,
            show_progress="hidden",
        )
        stop_btn.click(
            live_stop,
            inputs=None,
            outputs=[live_report_html, live_dashboard_img, live_pdf_download],
        )


def launch_app(**launch_overrides):
    """Launch with Gradio 5.x or 6.x compatible theme/css placement."""
    allowed = [
        str(APP_DIR.resolve()),
        str(OUTPUT_DIR.resolve()),
        str(core.OUTPUT_VIDEO_DIR.resolve()),
        str(core.OUTPUT_REPORT_DIR.resolve()),
        tempfile.gettempdir(),
    ]
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": 7860,
        "share": True,
        "ssr_mode": False,
        "allowed_paths": allowed,
    }
    launch_kwargs.update(launch_overrides)
    if GRADIO_MAJOR >= 6:
        launch_kwargs.setdefault("theme", APP_THEME)
        launch_kwargs.setdefault("css", CUSTOM_CSS)
    demo.queue().launch(**launch_kwargs)


if __name__ == "__main__":
    launch_app()
