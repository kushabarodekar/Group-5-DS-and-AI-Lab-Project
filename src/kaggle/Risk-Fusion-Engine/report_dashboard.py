"""
Colab-style dashboard and HTML report for the Driver Wellness AI deployment.
"""

from __future__ import annotations

import html
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from wellness_core import (
    DRIVER_ACTIVITY_RISK_MAP,
    FUSION_MODULE_NAME_TO_KEY,
    MODEL_REGISTRY,
    RISK_EVENT_SAFE_PREDICTIONS,
    RISK_EVENT_SEVERITY,
    PredictionResult,
)

RISK_COLOR_LOW = "#2ecc71"
RISK_COLOR_MEDIUM = "#f1c40f"
RISK_COLOR_HIGH = "#e67e22"
RISK_COLOR_CRITICAL = "#e74c3c"
RISK_COLOR_UNAVAILABLE = "#95a5a6"

_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "]+",
    flags=re.UNICODE,
)


def _plain_label(text: str) -> str:
    """Strip emoji for matplotlib labels (DejaVu Sans lacks emoji glyphs)."""
    cleaned = _EMOJI_RE.sub("", text or "").strip()
    return cleaned or str(text)


def _risk_color(risk_score: float) -> str:
    if risk_score < 0.25:
        return RISK_COLOR_LOW
    if risk_score < 0.50:
        return RISK_COLOR_MEDIUM
    if risk_score < 0.75:
        return RISK_COLOR_HIGH
    return RISK_COLOR_CRITICAL


def _risk_level_color(risk_level: str) -> str:
    level = (risk_level or "").lower()
    if "low" in level:
        return RISK_COLOR_LOW
    if "moderate" in level or "medium" in level or "caution" in level:
        return RISK_COLOR_MEDIUM
    if "high" in level:
        return RISK_COLOR_HIGH
    if "extreme" in level or "critical" in level:
        return RISK_COLOR_CRITICAL
    return RISK_COLOR_UNAVAILABLE


def _status_label(result: PredictionResult) -> str:
    if not result.is_error:
        return "OK"
    meta = result.metadata or {}
    exc_type = meta.get("exception_type")
    status = meta.get("status")
    if exc_type == "InsufficientBufferError" or status == "WARMING_UP":
        return "Video Too Short"
    if exc_type == "FaceNotDetectedError" or status == "UNAVAILABLE":
        return "No Face Detected"
    return "Unavailable"


def plot_wellness_score(ax: plt.Axes, overall_score: float, risk_level: str) -> None:
    zones = [
        (0, 25, RISK_COLOR_LOW),
        (25, 50, RISK_COLOR_MEDIUM),
        (50, 75, RISK_COLOR_HIGH),
        (75, 100, RISK_COLOR_CRITICAL),
    ]
    for start, end, color in zones:
        ax.barh(0, end - start, left=start, height=0.5, color=color, alpha=0.85)
    ax.axvline(overall_score, color="black", linewidth=3)
    ax.text(overall_score, 0.35, f"{overall_score:.1f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.4, 0.6)
    ax.set_yticks([])
    ax.set_xlabel("Driver Wellness Score (0 = Best · 100 = Worst)")
    ax.set_title(
        f"Overall Driver Wellness Score: {overall_score:.1f} / 100  —  {risk_level}",
        fontsize=13,
        fontweight="bold",
    )


def plot_summary_cards(ax: plt.Axes, results: List[PredictionResult]) -> None:
    ax.axis("off")
    num_modules = max(len(results), 1)
    card_width = 1.0 / num_modules
    for index, result in enumerate(results):
        x0 = index * card_width
        color = RISK_COLOR_UNAVAILABLE if result.is_error else _risk_color(result.risk_score)
        rect = mpatches.Rectangle(
            (x0 + 0.01, 0.05),
            card_width - 0.02,
            0.9,
            facecolor=color,
            alpha=0.25,
            edgecolor=color,
            linewidth=2,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(
            x0 + card_width / 2,
            0.78,
            _plain_label(result.module),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            wrap=True,
        )
        status_text = "Unavailable" if result.is_error else result.prediction
        ax.text(x0 + card_width / 2, 0.48, status_text, transform=ax.transAxes, ha="center", va="center", fontsize=10)
        detail_text = "—" if result.is_error else f"Risk {result.risk_score:.2f}  ·  Conf {result.confidence:.2f}"
        ax.text(x0 + card_width / 2, 0.20, detail_text, transform=ax.transAxes, ha="center", va="center", fontsize=8)
    ax.set_title("Module Summary", fontsize=13, fontweight="bold")


def plot_risk_breakdown(ax: plt.Axes, contributions: List[Dict[str, Any]]) -> None:
    modules = [_plain_label(contribution["module"]) for contribution in contributions]
    event_risks = [
        contribution.get("event_risk", contribution.get("weighted_contribution", 0.0))
        for contribution in contributions
    ]
    colors = [
        RISK_COLOR_UNAVAILABLE if not c["available"]
        else _risk_color(min(1.0, c.get("event_risk", 0.0) / 10.0))
        for c in contributions
    ]
    y_positions = np.arange(len(modules))
    ax.barh(y_positions, event_risks, color=colors)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(modules, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Event Risk R_i = severity × confidence")
    ax.set_title("Risk Breakdown by Module (Option A)", fontsize=13, fontweight="bold")
    for y, value in zip(y_positions, event_risks):
        ax.text(value, y, f" {value:.2f}", va="center", fontsize=8)


def plot_prediction_table(ax: plt.Axes, results: List[PredictionResult]) -> None:
    ax.axis("off")
    columns = ["Module", "Prediction", "Confidence", "Adapter Risk", "Severity w", "R_i", "Status"]
    rows = []
    for result in results:
        module_key = (result.metadata or {}).get("module_key")
        if module_key is None:
            module_key = FUSION_MODULE_NAME_TO_KEY.get(result.module)
        severity_w = 0.0
        event_risk = 0.0
        if not result.is_error and module_key is not None:
            safe_labels = RISK_EVENT_SAFE_PREDICTIONS.get(module_key, set())
            if result.prediction not in safe_labels:
                severity_w = float(RISK_EVENT_SEVERITY.get(module_key, {}).get(result.prediction, 0.0))
                event_risk = severity_w * float(result.confidence)
        rows.append(
            [
                _plain_label(result.module),
                "Unavailable" if result.is_error else result.prediction,
                "—" if result.is_error else f"{result.confidence:.2f}",
                "—" if result.is_error else f"{result.risk_score:.2f}",
                "—" if result.is_error else f"{severity_w:.1f}",
                "—" if result.is_error else f"{event_risk:.2f}",
                _status_label(result),
            ]
        )
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.7)
    ax.set_title("Prediction Table", fontsize=13, fontweight="bold")


def _module_display_name(module_key: str) -> str:
    if module_key in MODEL_REGISTRY:
        return _plain_label(MODEL_REGISTRY[module_key]["name"])
    return _plain_label(module_key.replace("_", " ").title())


def plot_streaming_timeline(ax: plt.Axes, streaming_result: Dict[str, Any]) -> None:
    score_timeline = streaming_result.get("score_timeline", [])
    module_timeline = streaming_result.get("module_timeline", {})
    plotted_any = False
    if score_timeline:
        times = [entry["timestamp_sec"] for entry in score_timeline]
        scores = [entry["overall_score"] / 100.0 for entry in score_timeline]
        ax.plot(times, scores, linewidth=2.2, label="Overall Wellness Score", color="black")
        plotted_any = True
    for module_key, series in module_timeline.items():
        if not series:
            continue
        times = [entry["timestamp_sec"] for entry in series]
        risks = [entry["risk_score"] for entry in series]
        ax.plot(times, risks, linewidth=1.3, alpha=0.85, label=_module_display_name(module_key))
        plotted_any = True
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Risk Signal (0 = Best · 1 = Worst)")
    ax.set_title("Timeline — Continuous Risk Signal Over the Video (Streaming)", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    if plotted_any:
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.text(0.5, 0.5, "No timeline data available", ha="center", va="center", transform=ax.transAxes)


def render_dashboard_figure(
    results: List[PredictionResult],
    fusion_result: Dict[str, Any],
    streaming_result: Optional[Dict[str, Any]] = None,
) -> plt.Figure:
    fig = plt.figure(figsize=(16, 18))
    grid = fig.add_gridspec(4, 2, height_ratios=[1.0, 1.3, 1.6, 1.6], hspace=0.55, wspace=0.3)
    ax_score = fig.add_subplot(grid[0, :])
    plot_wellness_score(ax_score, fusion_result["overall_score"], fusion_result["risk_level"])
    ax_cards = fig.add_subplot(grid[1, :])
    plot_summary_cards(ax_cards, results)
    ax_breakdown = fig.add_subplot(grid[2, 0])
    plot_risk_breakdown(ax_breakdown, fusion_result.get("contributions", []))
    ax_table = fig.add_subplot(grid[2, 1])
    plot_prediction_table(ax_table, results)
    ax_timeline = fig.add_subplot(grid[3, :])
    if streaming_result and streaming_result.get("score_timeline"):
        plot_streaming_timeline(ax_timeline, streaming_result)
    else:
        plot_streaming_timeline(ax_timeline, {"score_timeline": [], "module_timeline": {}})
    fig.suptitle("Driver Wellness AI — Dashboard", fontsize=17, fontweight="bold")
    fig.subplots_adjust(top=0.96, hspace=0.55, wspace=0.3)
    return fig


def save_dashboard_png(
    analysis: Dict[str, Any],
    output_path: Optional[str] = None,
) -> str:
    fusion = analysis.get("fusion_result") or {}
    if not fusion:
        fusion = {"overall_score": 0.0, "risk_level": "N/A", "contributions": []}
    fig = render_dashboard_figure(
        analysis.get("prediction_results", []),
        fusion,
        streaming_result=analysis,
    )
    path = output_path or str(Path(tempfile.gettempdir()) / f"dashboard_{id(analysis)}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_dashboard_pdf(
    analysis: Dict[str, Any],
    output_path: Optional[str] = None,
) -> str:
    fusion = analysis.get("fusion_result") or {}
    if not fusion:
        fusion = {"overall_score": 0.0, "risk_level": "N/A", "contributions": []}
    path = output_path or str(Path(tempfile.gettempdir()) / f"dashboard_{id(analysis)}.pdf")
    fig = render_dashboard_figure(
        analysis.get("prediction_results", []),
        fusion,
        streaming_result=analysis,
    )
    with PdfPages(path) as pdf:
        pdf.savefig(fig, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def build_html_report(analysis: Dict[str, Any], summary: Dict[str, Any]) -> str:
    fusion = analysis.get("fusion_result") or {}
    overall = float(fusion.get("overall_score", 0.0))
    risk_level = html.escape(str(fusion.get("risk_level", "N/A")))
    badge_color = _risk_level_color(str(fusion.get("risk_level", "")))

    cards_html = []
    for result in analysis.get("prediction_results", []):
        color = RISK_COLOR_UNAVAILABLE if result.is_error else _risk_color(result.risk_score)
        pred = html.escape("Unavailable" if result.is_error else str(result.prediction))
        module = html.escape(result.module)
        conf = "—" if result.is_error else f"{result.confidence:.2f}"
        risk = "—" if result.is_error else f"{result.risk_score:.2f}"
        cards_html.append(
            f"""
            <div class="module-card" style="border-color:{color}; background:{color}22;">
              <div class="module-name">{module}</div>
              <div class="module-pred">{pred}</div>
              <div class="module-meta">Risk {risk} · Conf {conf}</div>
            </div>
            """
        )

    table_rows = []
    for contrib in fusion.get("contributions", []):
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(contrib.get('module', '')))}</td>"
            f"<td>{html.escape(str(contrib.get('prediction', '')))}</td>"
            f"<td>{contrib.get('confidence', 0):.2f}</td>"
            f"<td>{contrib.get('event_risk', 0):.2f}</td>"
            f"<td>{contrib.get('severity_weight', 0):.1f}</td>"
            f"<td>{'OK' if contrib.get('available') else 'Unavailable'}</td>"
            "</tr>"
        )

    stats_items = "".join(
        f"<li><strong>{html.escape(str(k))}:</strong> {html.escape(str(v))}</li>"
        for k, v in summary.items()
        if not str(k).strip().startswith("•")
    )

    gauge_pct = max(0.0, min(100.0, overall))
    return f"""
    <div class="dw-report">
      <div class="score-header">
        <h2>Overall Driver Wellness Score</h2>
        <div class="score-value">{overall:.1f} <span>/ 100</span></div>
        <span class="risk-badge" style="background:{badge_color};">{risk_level}</span>
      </div>
      <div class="gauge-track">
        <div class="gauge-zone low"></div>
        <div class="gauge-zone medium"></div>
        <div class="gauge-zone high"></div>
        <div class="gauge-zone critical"></div>
        <div class="gauge-marker" style="left:{gauge_pct}%;"></div>
      </div>
      <p class="gauge-caption">Driver Wellness Score (0 = Best; 100 = Worst)</p>
      <h3>Module Summary</h3>
      <div class="module-grid">{''.join(cards_html)}</div>
      <h3>Fusion Breakdown</h3>
      <table class="fusion-table">
        <thead>
          <tr><th>Module</th><th>Prediction</th><th>Confidence</th><th>R_i</th><th>Severity</th><th>Status</th></tr>
        </thead>
        <tbody>{''.join(table_rows)}</tbody>
      </table>
      <details class="session-details">
        <summary>Session statistics</summary>
        <ul>{stats_items}</ul>
      </details>
    </div>
    """


def build_live_status_html(fusion: Dict[str, Any], frame_count: int, fps: float = 0.0) -> str:
    """Compact Colab-style live status panel for streaming webcam updates."""
    overall = float(fusion.get("overall_score", 0.0))
    risk_level = html.escape(str(fusion.get("risk_level", "N/A")))
    badge_color = _risk_level_color(str(fusion.get("risk_level", "")))
    gauge_pct = max(0.0, min(100.0, overall))

    module_lines = []
    for contrib in fusion.get("contributions", []):
        name = html.escape(str(contrib.get("module", "?")))
        pred = html.escape(str(contrib.get("prediction", "—")))
        risk = contrib.get("event_risk", 0.0)
        module_lines.append(f"<li><strong>{name}:</strong> {pred} <span>(risk {risk:.2f})</span></li>")

    warming = fusion.get("modules_warming_up") or []
    warming_text = ""
    if warming:
        warming_text = (
            "<p class='live-warming'><strong>Warming up:</strong> "
            + html.escape(", ".join(warming))
            + "</p>"
        )

    return f"""
    <div class="dw-report dw-live-status">
      <div class="score-header">
        <div class="score-value">{overall:.1f} <span>/ 100</span></div>
        <span class="risk-badge" style="background:{badge_color};">{risk_level}</span>
      </div>
      <div class="gauge-track">
        <div class="gauge-zone low"></div>
        <div class="gauge-zone medium"></div>
        <div class="gauge-zone high"></div>
        <div class="gauge-zone critical"></div>
        <div class="gauge-marker" style="left:{gauge_pct}%;"></div>
      </div>
      <p class="gauge-caption">Frames: {frame_count} · FPS: {fps:.1f}</p>
      {warming_text}
      <ul class="live-module-list">{''.join(module_lines)}</ul>
    </div>
    """


def build_video_player_html(video_path: Optional[str]) -> str:
    """HTML5 fallback player — works when gr.Video preview fails but download succeeds."""
    if not video_path:
        return "<p class='status-pill'>Annotated video will appear here after analysis.</p>"
    abs_path = html.escape(str(Path(video_path).resolve()))
    return f"""
    <div class="video-wrap">
      <video controls autoplay playsinline style="width:100%;max-height:520px;border-radius:12px;background:#111;"
             src="/gradio_api/file={abs_path}"></video>
      <p style="margin-top:0.5rem;font-size:0.9rem;">
        <a href="/gradio_api/file={abs_path}" download="annotated_output.mp4">Download annotated video</a>
      </p>
    </div>
    """


def build_report_assets(analysis: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, str]:
    from wellness_core import OUTPUT_REPORT_DIR

    stamp = str(abs(hash(str(analysis.get("frames_processed", 0)))))[:10]
    base = OUTPUT_REPORT_DIR / f"dw_report_{stamp}"
    png_path = save_dashboard_png(analysis, str(base.with_suffix(".png")))
    pdf_path = save_dashboard_pdf(analysis, str(base.with_suffix(".pdf")))
    return {
        "html": build_html_report(analysis, summary),
        "dashboard_png": png_path,
        "dashboard_pdf": pdf_path,
    }
