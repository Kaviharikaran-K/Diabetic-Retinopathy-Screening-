#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app_interface.py
================
Explainable AI Diabetic Retinopathy screening -- Streamlit clinical dashboard.

Dark-themed dual-eye diagnostic terminal:

  * Left Eye (OS) and Right Eye (OD) processed **simultaneously** by the
    core engine (``pipeline_engine.run_case``).
  * Side-by-side visual Explainability panels: raw fundus, Grad-CAM /
    saliency heatmap, attention-rollout map and lesion annotation.
  * Diagnostic breakdown: IQA scores, ICDR level 0-4, confidence,
    uncertainty flag and lesion metrics (MAs, haemorrhages, exudates, NV).
  * Human-in-the-loop clinical actions per eye:
        [Confirm AI Grade] [Modify Grade] [Mark Ungradable]
        [Confirm Referral] [Save Doctor Review]
  * One-click export of the risk-priority queue + latency profile consumed by
    the MATLAB SimEvents telemedicine simulator.

Run with:
    streamlit run app_interface.py
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

import pipeline_engine as pe

# ---------------------------------------------------------------------------
# Application configuration.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
EXPORTS_DIR = BASE_DIR / "exports"
REVIEWS_FILE = EXPORTS_DIR / "clinical_reviews.jsonl"
EXPORTS_DIR.mkdir(exist_ok=True)

GRADE_OPTIONS = {"-1": "Ungradable", **{str(i): f"ICDR {i}" for i in range(5)}}


# ---------------------------------------------------------------------------
# Theme CSS -- mimics a clinical diagnostic terminal.
# ---------------------------------------------------------------------------
def inject_theme_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0b0f17;
            --panel: #111826;
            --panel2: #161f31;
            --accent: #38bdf8;
            --accent2: #22c55e;
            --danger: #ef4444;
            --warn: #f59e0b;
            --text: #dbe4f0;
        }
        .stApp { background: var(--bg); }
        html, body, [class*="css"] {
            font-family: "JetBrains Mono", "Consolas", "Roboto Mono", monospace;
        }
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
        h1, h2, h3 { color: var(--text); letter-spacing: .5px; }
        div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid #22324d;
            border-radius: 10px;
            padding: 10px 14px;
        }
        div[data-testid="stMetricLabel"] { color: #7c8db5; font-size: .72rem; }
        div[data-testid="stMetricValue"] { color: var(--text); }
        button[kind="primary"] {
            background: linear-gradient(180deg,#1d4ed8,#173a8f);
            border: 1px solid #3b6cd4;
            color: #fff;
            font-weight: 600;
        }
        .term-banner {
            background: linear-gradient(90deg,#0d1524,#122033);
            border: 1px solid #1e3352;
            border-radius: 12px;
            padding: 14px 18px;
            color: var(--text);
            margin-bottom: 10px;
        }
        .term-banner .title { font-size: 1.15rem; font-weight: 700; color: var(--accent); }
        .term-banner .subt { color: #8ea3c8; font-size: .8rem; margin-top: 4px; }
        .eyed-head {
            color: var(--accent2); font-weight: 700; letter-spacing: 1px;
            margin-bottom: 2px;
        }
        .badge-gradeable { color: var(--accent2); }
        .badge-ungradable { color: var(--danger); }
        .rgn { color: #8ad0ff; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar: image siblings.
# ---------------------------------------------------------------------------
def sidebar_inputs() -> dict:
    st.sidebar.header("SCREENING INPUT")
    mode = st.sidebar.radio(
        "Input source", ("Upload fundus images", "Synthetic demo (no files)")
    )

    od_bytes, os_bytes = None, None
    if mode.startswith("Upload"):
        od = st.sidebar.file_uploader(
            "Right eye image (OD)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            key="upl_od",
        )
        os_ = st.sidebar.file_uploader(
            "Left eye image (OS)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            key="upl_os",
        )
        if od is not None:
            od_bytes = od.getvalue()
        if os_ is not None:
            os_bytes = os_.getvalue()
    else:
        od_sev = st.sidebar.selectbox(
            "OD severity (right)", ["proliferative", "severe", "moderate", "mild", "normal"],
            index=0,
        )
        os_sev = st.sidebar.selectbox(
            "OS severity (left)", ["moderate", "severe", "mild", "normal", "proliferative"],
            index=0,
        )
        st.session_state["demo_sev"] = (od_sev, os_sev)

    parallel = st.sidebar.checkbox("Process both eyes in parallel", value=True)
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Engine: INT8 MobileViT-XS + DRNet13 fused with ICDR staging profiler."
    )
    return {"od_bytes": od_bytes, "os_bytes": os_bytes, "parallel": parallel}


# ---------------------------------------------------------------------------
# Decode uploaded bytes (or generate demo fundi).
# ---------------------------------------------------------------------------
def resolve_images(ctx: dict) -> tuple:
    if ctx["od_bytes"] and ctx["os_bytes"]:
        od = cv2.imdecode(np.frombuffer(ctx["od_bytes"], np.uint8), cv2.IMREAD_COLOR)
        os_ = cv2.imdecode(np.frombuffer(ctx["os_bytes"], np.uint8), cv2.IMREAD_COLOR)
        if od is None or os_ is None:
            st.error("One or both uploaded files could not be decoded.")
            st.stop()
        return od, os_

    od_sev, os_sev = st.session_state.get("demo_sev", ("proliferative", "moderate"))
    return (
        pe.generate_synthetic_fundus("OD", od_sev, seed=101),
        pe.generate_synthetic_fundus("OS", os_sev, seed=202),
    )


def cache_key_for(*arrays: np.ndarray, parallel: bool) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(a.tobytes())
    h.update(str(parallel).encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Run the engine (cached in session state; re-run button forces refresh).
# ---------------------------------------------------------------------------
def run_screening() -> pe.CaseReport:
    key = cache_key_for(
        st.session_state["od_img"],
        st.session_state["os_img"],
        parallel=st.session_state.get("parallel", True),
    )
    machine_key = f"case_{key}"
    if machine_key in st.session_state:
        return st.session_state[machine_key]

    t0 = time.perf_counter()
    with st.spinner("Running dual-eye XAI pipeline (both eyes in parallel)..."):
        case = pe.run_case(
            st.session_state["od_img"],
            st.session_state["os_img"],
            use_parallel=True,
        )
    case._wall_s = time.perf_counter() - t0  # type: ignore[attr-defined]
    st.session_state[machine_key] = case
    return case


# ---------------------------------------------------------------------------
# Per-eye visual panels.
# ---------------------------------------------------------------------------
def render_image_row(eye: pe.EyeReport) -> None:
    t1, t2, t3 = st.tabs(["Fundus", "Grad-CAM / Saliency", "Lesion map"])
    if eye.fundus_rgb is not None:
        with t1:
            st.image(eye.fundus_rgb, use_container_width=True)
        with t2:
            if eye.heatmap_overlay is not None:
                st.image(eye.heatmap_overlay, use_container_width=True)
                if eye.attention_rollout is not None:
                    st.caption("Overlay = Grad-CAM (or feature saliency when the "
                               "neural head lacks weights); attention rollout available "
                               "for MobileViT-XS.")
            else:
                st.caption("No heatmap produced.")
        with t3:
            if eye.lesion_annotation is not None:
                st.image(eye.lesion_annotation, use_container_width=True)
                st.caption("Red=MA  Green=HEM  Blue=EX  Grey=optic disc  White=fovea")
            else:
                st.caption("No annotation produced.")
    else:
        st.caption("Image unavailable for this eye.")


def render_diagnostic_breakdown(eye: pe.EyeReport) -> None:
    if not eye.gradeable:
        st.warning("IQA gate: image rejected for grading.")
        for r in eye.reasons:
            st.write(f"- {r}")
        return

    c = st.columns(4)
    c[0].metric("ICDR Grade", f"{eye.icdr_grade}/4")
    c[1].metric("Confidence", f"{eye.confidence * 100:.1f}%")
    c[2].metric("Uncertainty", "FLAGGED" if eye.uncertainty_flag else "ok")
    c[3].metric("Priority", f"P{eye.priority}")

    c = st.columns(4)
    c[0].metric("Microaneurysms", len(eye.lesions.microaneurysms))
    c[1].metric("Haemorrhages", len(eye.lesions.hemorrhages))
    c[2].metric("Hard Exudates", len(eye.lesions.hard_exudates))
    c[3].metric("Neovascularisation", "YES" if eye.vessels.nv_flag else "No")

    st.markdown("**Staging rationale**")
    for r in eye.reasons:
        st.write(f"- {r}")

    with st.expander("Image Quality Assessment detail"):
        st.write(f"Sharpness (variance of Laplacian):        {eye.iqa.sharpness_variance:.2f}")
        st.write(f"Illumination (mean luminance):            {eye.iqa.luminance_mean:.2f}")
        st.write(f"Contrast (green-channel std):             {eye.iqa.contrast_std:.2f}")
        st.write(f"Field coverage:                           {eye.iqa.coverage_ratio:.3f}")
        st.write(f"Composite quality score:                  {eye.iqa.quality_score:.1f}")
        st.progress(min(1.0, eye.iqa.quality_score / 100.0))


# ---------------------------------------------------------------------------
# Human-in-the-loop clinical actions.
# ---------------------------------------------------------------------------
def initialise_review_state(eye_name: str) -> None:
    key = f"review_{eye_name}"
    if key not in st.session_state:
        st.session_state[key] = {
            "confirmed_grade": None,
            "modified_grade": None,
            "marked_ungradable": False,
            "confirmed_referral": False,
            "saved": False,
        }


def persist_review(eye_name: str, review: dict) -> None:
    line = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "eye": eye_name,
        **review,
    }
    with open(REVIEWS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


def render_clinical_actions(eye_name: str, eye: pe.EyeReport) -> None:
    key = f"review_{eye_name}"
    initialise_review_state(eye_name)
    rev = st.session_state[key]

    g = ["Confirm AI Grade", "Modify Grade", "Mark Ungradable", "Confirm Referral"]
    c1, c2, c3, c4 = st.columns(4)

    if c1.button(g[0], key=f"btn_{eye_name}_confirm") and eye.gradeable:
        rev["confirmed_grade"] = eye.icdr_grade
        rev["confirmed_referral"] = eye.refer

    if c2.button(g[1], key=f"btn_{eye_name}_modify"):
        rev["modify_open"] = not rev.get("modify_open", False)

    if c3.button(g[2], key=f"btn_{eye_name}_ungradable"):
        rev["marked_ungradable"] = True
        rev["confirmed_grade"] = None

    if c4.button(g[3], key=f"btn_{eye_name}_refer"):
        rev["confirmed_referral"] = True

    if rev.get("modify_open"):
        st.selectbox(
            "Override AI grade",
            options=list(GRADE_OPTIONS.keys()),
            format_func=lambda k: GRADE_OPTIONS[k],
            index=list(GRADE_OPTIONS.keys()).index(str(eye.icdr_grade)) if eye.icdr_grade >= 0 else 0,
            key=f"sel_{eye_name}_grade",
        )

    save_col, status_col = st.columns([1, 3])
    if save_col.button("Save Doctor Review", key=f"btn_{eye_name}_save"):
        chosen_key = st.session_state.get(f"sel_{eye_name}_grade")
        review = {
            "ai_grade": eye.icdr_grade if eye.gradeable else -1,
            "doctor_grade": (
                GRADE_OPTIONS[chosen_key] if chosen_key is not None else None
            ),
            "confirmed_grade": rev["confirmed_grade"],
            "marked_ungradable": rev["marked_ungradable"],
            "confirmed_referral": rev["confirmed_referral"],
            "final_referral": rev["confirmed_referral"] or eye.refer,
            "iqa_score": eye.iqa.quality_score,
        }
        persist_review(eye_name, review)
        rev["saved"] = True

    if rev["saved"]:
        status_col.success("Review saved.")
    else:
        status_col.caption("Doctor review not yet recorded.")


def render_review_history() -> None:
    with st.expander("Doctor review history (human-in-the-loop log)"):
        if REVIEWS_FILE.exists():
            entries = []
            for line in REVIEWS_FILE.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            if entries:
                st.table(entries[::-1][:20])
            else:
                st.caption("No reviews logged yet.")
        else:
            st.caption("No reviews logged yet.")


# ---------------------------------------------------------------------------
# MATLAB hand-off.
# ---------------------------------------------------------------------------
def matlab_handoff(case: pe.CaseReport) -> None:
    st.sidebar.markdown("---")
    st.sidebar.header("SIMULATION HAND-OFF")
    if st.sidebar.button("Export for MATLAB SimEvents", use_container_width=True):
        paths = pe.export_metrics(case, str(EXPORTS_DIR))
        st.sidebar.success("Exported:")
        for v in paths.values():
            st.sidebar.caption(v)
        st.sidebar.caption("Now run `telemedicine_sim` in MATLAB.")


# ---------------------------------------------------------------------------
# Main dashboard.
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="DR-XAI Screening Terminal", layout="wide")
    inject_theme_css()

    st.markdown(
        """
        <div class="term-banner">
          <div class="title">EXPLAINABLE AI &middot; DIABETIC RETINOPATHY SCREENING</div>
          <div class="subt">Offline dual-eye pipeline &middot; OD/OS simultaneous &middot;
          INT8 MobileViT-XS + DRNet13 &middot; Grad-CAM XAI &middot; ICDR 0-4 &middot;
          MATLAB SimEvents link</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ctx = sidebar_inputs()

    if st.button("RUN SCREENING", type="primary", use_container_width=True):
        od_img, os_img = resolve_images(ctx)
        st.session_state["od_img"] = od_img
        st.session_state["os_img"] = os_img
        st.session_state.setdefault("parallel", ctx["parallel"])

    if "od_img" not in st.session_state or "os_img" not in st.session_state:
        st.info("Load both eyes (or enable the synthetic demo) and press "
                "RUN SCREENING to start the interpretable pipeline.")
        return

    case = run_screening()

    # ---- engine status strip ------------------------------------------------
    wall = getattr(case, "_wall_s", 0.0)
    c = st.columns(5)
    c[0].metric("Engine", "ON-LINE")
    c[1].metric("Neural head", "INT8-quantized" if case.quantized_int8 else "Profiler")
    c[2].metric("Encounter priority", f"P{case.overall_priority}")
    c[3].metric("Referral", "REQUIRED" if case.overall_refer else "None")
    c[4].metric("Runtime", f"{wall:.2f}s")

    # ---- dual-eye grid --------------------------------------------------------
    col_l, col_r = st.columns(2, gap="medium")

    with col_r:
        st.markdown('<div class="eyed-head">RIGHT EYE :: OD</div>',
                    unsafe_allow_html=True)
        render_image_row(case.od)
        render_diagnostic_breakdown(case.od)
        render_clinical_actions("OD", case.od)

    with col_l:
        st.markdown('<div class="eyed-head">LEFT EYE :: OS</div>',
                    unsafe_allow_html=True)
        render_image_row(case.os)
        render_diagnostic_breakdown(case.os)
        render_clinical_actions("OS", case.os)

    st.markdown("---")
    render_review_history()
    matlab_handoff(case)


if __name__ == "__main__":
    main()