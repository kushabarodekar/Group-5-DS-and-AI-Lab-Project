"""
Driver Wellness AI — Hugging Face Spaces entry point (Gradio).

This file is intentionally thin. All model/adapter/fusion/orchestrator logic
lives in `wellness_core.py` (a faithful port of the Colab notebook, with the
updated Option-A exponential risk fusion + score smoothing). `app.py` only:

  1. builds the module manager once at startup,
  2. exposes `analyze_video(path)` for the recorded-video tab, and
  3. exposes a live webcam streaming tab that reuses the same orchestrator,
     one browser-streamed frame at a time.

Public surface used from wellness_core.py:
    build_manager()                         -> loads all 5 models, returns manager
    run_recorded_video(manager, video_path) -> (annotated_video_path, summary_dict)
    start_live_session(fps=None)            -> reset orchestrator for a new live run
    process_live_frame(frame_rgb, fps=None) -> (annotated_rgb, summary_text)
    stop_live_session()                     -> final summary string
"""

import traceback

import gradio as gr

# ----------------------------------------------------------------------
# `spaces` is only present on Hugging Face ZeroGPU hardware. Guard the
# import so the app also runs on a plain CPU/GPU Space (or locally) where
# the package may be absent — the decorator then becomes a no-op.
# ----------------------------------------------------------------------
try:
    import spaces  # noqa: F401

    def gpu(duration=120):
        return spaces.GPU(duration=duration)
except Exception:  # pragma: no cover
    def gpu(duration=120):
        def _wrap(fn):
            return fn
        return _wrap

# ----------------------------------------------------------------------
# Optional: pull weights from a separate HF model repo instead of shipping
# them inside this Space. Uncomment and set MODEL_REPO to use this path.
# ----------------------------------------------------------------------
# import os
# from huggingface_hub import hf_hub_download
# MODEL_REPO = "your-username/driver-wellness-weights"
# os.makedirs("models", exist_ok=True)
# for fname in [
#     "Video_Fatigue.pth", "Landmark_Fatigue.pt", "Driver_Activity.pth",
#     "Smoking_And_Drinking.pt", "SeatBelt_And_Phone.pt",
#     "m4_normalization_stats_ws45.csv", "face_landmarker.task",
# ]:
#     hf_hub_download(repo_id=MODEL_REPO, filename=fname, local_dir="models")

# ----------------------------------------------------------------------
# Build the manager ONCE (loading 5 models is expensive — never per request)
# ----------------------------------------------------------------------
import wellness_core as core

print("Loading models... (first startup can take a minute)")
MANAGER = core.build_manager()
print("Models loaded.")


# ======================================================================
# Recorded-video tab
# ======================================================================
@gpu(duration=120)
def analyze_video(video_path):
    if not video_path:
        return None, "Please upload a video first."
    try:
        annotated_path, summary = core.run_recorded_video(MANAGER, video_path)
        lines = [f"{k}: {v}" for k, v in summary.items()]
        return annotated_path, "\n".join(lines)
    except Exception:
        return None, "Error during analysis:\n" + traceback.format_exc()


# ======================================================================
# Live webcam tab (browser streams frames -> same orchestrator)
# ======================================================================
def live_start():
    """Reset the orchestrator for a fresh live session."""
    core.start_live_session()
    return None, "Live session started — grant camera access and stay in frame."


def live_stream(frame):
    """Handle ONE streamed webcam frame; return (annotated_rgb, summary_text)."""
    if frame is None:
        return None, "Waiting for webcam frames..."
    try:
        return core.process_live_frame(frame)
    except Exception:
        return frame, "Error during live analysis:\n" + traceback.format_exc()


def live_stop():
    """Finalise the live session and return the fused summary."""
    try:
        return core.stop_live_session()
    except Exception:
        return "Error finalising session:\n" + traceback.format_exc()


with gr.Blocks(title="Driver Wellness AI") as demo:
    gr.Markdown(
        "# 🚗 Driver Wellness AI\n"
        "Five models (video fatigue, landmark fatigue, driver activity, "
        "seat-belt/phone, smoking/drinking) fused into one wellness/risk score "
        "using the Common Driver Risk Score Framework (Option A, exponential)."
    )

    with gr.Tab("📹 Recorded video"):
        gr.Markdown("Upload a short driving clip to get a fused, annotated analysis.")
        with gr.Row():
            with gr.Column():
                inp = gr.Video(label="Driving clip", sources=["upload"])
                btn = gr.Button("Analyze", variant="primary")
            with gr.Column():
                out_video = gr.Video(label="Annotated output")
                out_text = gr.Textbox(label="Session summary", lines=14)
        btn.click(analyze_video, inputs=inp, outputs=[out_video, out_text])

    with gr.Tab("🔴 Live webcam"):
        gr.Markdown(
            "Stream your webcam for a real-time fused analysis. Click **Start**, "
            "allow camera access, and the annotated feed + live risk score update "
            "continuously. Click **Finish** for the session summary.\n\n"
            "> On CPU Spaces this runs slowly; upgrade to GPU hardware for smoother "
            "real-time performance."
        )
        with gr.Row():
            with gr.Column():
                cam = gr.Image(
                    label="Webcam",
                    sources=["webcam"],
                    streaming=True,
                    type="numpy",
                )
                with gr.Row():
                    start_btn = gr.Button("▶ Start / Reset session", variant="primary")
                    stop_btn = gr.Button("⏹ Finish & summarize")
            with gr.Column():
                live_out = gr.Image(label="Annotated live feed", type="numpy")
                live_text = gr.Textbox(label="Live status", lines=12)
        final_text = gr.Textbox(label="Session summary", lines=10)

        start_btn.click(live_start, inputs=None, outputs=[live_out, live_text])
        # Stream each captured frame through the orchestrator. concurrency_limit=1
        # keeps the shared orchestrator state serialized (frames processed in order).
        cam.stream(
            live_stream,
            inputs=[cam],
            outputs=[live_out, live_text],
            stream_every=0.1,
            concurrency_limit=1,
            show_progress="hidden",
        )
        stop_btn.click(live_stop, inputs=None, outputs=[final_text])


if __name__ == "__main__":
    demo.queue().launch(ssr_mode=False)
