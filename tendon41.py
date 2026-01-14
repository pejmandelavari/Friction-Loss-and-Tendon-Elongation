# app_pt_2d_final.py
# Streamlit app: 2D PT tendon (C1) + friction (left/right/double) + theoretical elongation + professional PDF report
# - User can upload a logo (PNG/JPG) for the report header
# - PDF includes header/footer, page number, disclaimer, inputs, results, tables, and figures
# - Fixes ReportLab LayoutError by scaling images to fit the page frame safely

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from io import BytesIO
from datetime import datetime

# ReportLab
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.utils import ImageReader

# -----------------------------
# Streamlit page
# -----------------------------
st.set_page_config(
    page_title="Post-Tensioning Tendon Analysis System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stMetric:hover {
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    h1 {
        color: #1f77b4;
        font-weight: 600;
        padding-bottom: 1rem;
        border-bottom: 2px solid #e0e0e0;
    }
    h2 {
        color: #2c3e50;
        font-weight: 500;
        margin-top: 1.5rem;
    }
    h3 {
        color: #34495e;
        font-weight: 500;
    }
    div[data-testid="stDataFrameResizable"] {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# Math / geometry utilities
# -----------------------------
def hermite_eval(x0, z0, m0, x1, z1, m1, xq):
    """Cubic Hermite z(x) on [x0,x1] with z(x0)=z0,z'(x0)=m0,z(x1)=z1,z'(x1)=m1."""
    xq = np.asarray(xq, dtype=float)
    L = x1 - x0
    if L <= 0:
        raise ValueError("Segment length must be positive.")
    t = (xq - x0) / L

    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2

    return h00*z0 + h10*(L*m0) + h01*z1 + h11*(L*m1)


def compute_theta_from_profile(x, z):
    """
    Cumulative absolute change in tangent angle (theta) at nodes.
    theta[0]=0, theta[i]=sum |delta angle| up to node i.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)

    dx = np.diff(x)
    dz = np.diff(z)

    slope = np.zeros_like(dx)
    mask = np.abs(dx) > 1e-12
    slope[mask] = dz[mask] / dx[mask]

    ang = np.arctan(slope)  # angle per segment
    dtheta = np.abs(np.diff(ang, prepend=ang[0]))

    theta_nodes = np.zeros_like(x)
    theta_nodes[1:] = np.cumsum(dtheta)
    return theta_nodes


def friction_force(P0, mu, k, x, theta):
    """P = P0 * exp(-(mu*theta + k*x)) ; x must be measured from jacking end."""
    x = np.asarray(x, dtype=float)
    theta = np.asarray(theta, dtype=float)
    return P0 * np.exp(-(mu*theta + k*x))


def true_tendon_length(x, z):
    """Polyline length along the actual (x,z) profile."""
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    dx = np.diff(x)
    dz = np.diff(z)
    return float(np.sum(np.sqrt(dx**2 + dz**2)))


def theoretical_elongation_mm(P_kN, x_m, A_mm2, E_MPa):
    P_kN = np.asarray(P_kN, dtype=float)
    x_m = np.asarray(x_m, dtype=float)

    A_m2 = float(A_mm2) * 1e-6
    E_Pa = float(E_MPa) * 1e6
    if A_m2 <= 0 or E_Pa <= 0:
        return np.nan

    y = P_kN * 1000.0  # N

    # NumPy 2.x compatible + backward compatible
    if hasattr(np, "trapezoid"):
        area_under = np.trapezoid(y, x_m)  # N*m
    else:
        area_under = np.trapz(y, x_m)      # N*m

    delta_m = area_under / (A_m2 * E_Pa)
    return float(delta_m * 1000.0)  # mm


def spans_to_support_stations(span_lengths):
    s = [0.0]
    for L in span_lengths:
        s.append(s[-1] + float(L))
    return np.array(s, dtype=float)


def upsert_point(ctrl, typ, x_new, z_new, slope_new, eps=1e-6):
    """
    Insert or update a control point by x (within eps).
    Keep highest priority type among (SUP > LOW > INF > END).
    """
    ctrl = ctrl.copy()
    priority = {"SUP": 0, "LOW": 1, "INF": 2, "END": 3}

    mask = (ctrl["x"] - x_new).abs() < eps
    if mask.any():
        idx = ctrl.index[mask][0]
        old_typ = str(ctrl.loc[idx, "type"])
        if priority.get(typ, 99) < priority.get(old_typ, 99):
            ctrl.loc[idx, "type"] = typ
        ctrl.loc[idx, "z"] = float(z_new)
        ctrl.loc[idx, "slope"] = float(slope_new)
        ctrl.loc[idx, "slope_mode"] = "forced"
        return ctrl.sort_values("x").reset_index(drop=True)

    add = pd.DataFrame([{
        "type": typ,
        "x": float(x_new),
        "z": float(z_new),
        "slope": float(slope_new),
        "slope_mode": "forced"
    }])
    ctrl = pd.concat([ctrl, add], ignore_index=True).sort_values("x").reset_index(drop=True)
    return ctrl


def build_base_control_points(span_df, support_h, pier_df, zero_slope_low=True, min_gap=0.25):
    """
    Base control points:
      - SUP at all supports: interior SUP slope forced 0, end SUP slope auto (nan)
      - LOW per span: slope forced 0 if option, else auto
      - INF around interior supports using d_before/d_after, with z by interpolation.
    Z inputs are heights from soffit.
    """
    n_spans = len(span_df)
    span_lengths = span_df["L (m)"].astype(float).to_list()
    supports_x = spans_to_support_stations(span_lengths)
    Ltot = float(supports_x[-1])

    pts = []

    # Supports
    for j, xj in enumerate(supports_x):
        zj = float(support_h[j])
        if (j != 0) and (j != n_spans):
            slope = 0.0
            mode = "forced"
        else:
            slope = np.nan
            mode = "auto"
        pts.append(("SUP", float(xj), zj, slope, mode))

    # LOW points
    low_xz = []
    for i in range(n_spans):
        L = float(span_df.loc[i, "L (m)"])
        alpha = float(span_df.loc[i, "alpha (low point as fraction of L)"])
        x0 = float(supports_x[i])
        x_low = x0 + alpha * L
        z_low = float(span_df.loc[i, "low_h_from_soffit (m)"])
        low_xz.append((x_low, z_low))

        if zero_slope_low:
            pts.append(("LOW", float(x_low), z_low, 0.0, "forced"))
        else:
            pts.append(("LOW", float(x_low), z_low, np.nan, "auto"))

    # INF control points around interior supports
    for pier_idx in range(1, n_spans):
        x_p = float(supports_x[pier_idx])
        z_p = float(support_h[pier_idx])

        d_before = float(pier_df.loc[pier_idx, "d_before (m)"])
        d_after = float(pier_df.loc[pier_idx, "d_after (m)"])

        x_low_L, z_low_L = low_xz[pier_idx - 1]
        x_low_R, z_low_R = low_xz[pier_idx]

        x_inf_L = x_p - d_before
        x_inf_R = x_p + d_after

        # clamp
        x_inf_L = max(min(x_inf_L, x_p - min_gap), x_low_L + min_gap)
        x_inf_R = min(max(x_inf_R, x_p + min_gap), x_low_R - min_gap)

        # validity
        if not (x_low_L + min_gap < x_inf_L < x_p - min_gap):
            continue
        if not (x_p + min_gap < x_inf_R < x_low_R - min_gap):
            continue

        # interpolate z along (LOW_L -> SUP) and (SUP -> LOW_R)
        tL = (x_inf_L - x_low_L) / max((x_p - x_low_L), 1e-12)
        z_inf_L = z_low_L + tL * (z_p - z_low_L)

        tR = (x_inf_R - x_p) / max((x_low_R - x_p), 1e-12)
        z_inf_R = z_p + tR * (z_low_R - z_p)

        pts.append(("INF", float(x_inf_L), float(z_inf_L), np.nan, "auto"))
        pts.append(("INF", float(x_inf_R), float(z_inf_R), np.nan, "auto"))

    ctrl = pd.DataFrame(pts, columns=["type", "x", "z", "slope", "slope_mode"]).sort_values("x").reset_index(drop=True)

    # merge near-duplicates by x
    priority = {"SUP": 0, "LOW": 1, "INF": 2, "END": 3}
    eps = 1e-6
    merged = []
    i = 0
    while i < len(ctrl):
        group = [ctrl.iloc[i].to_dict()]
        j = i + 1
        while j < len(ctrl) and abs(float(ctrl.loc[j, "x"]) - float(ctrl.loc[i, "x"])) < eps:
            group.append(ctrl.iloc[j].to_dict())
            j += 1
        group.sort(key=lambda d: priority.get(d["type"], 99))
        merged.append(group[0])
        i = j

    ctrl = pd.DataFrame(merged).sort_values("x").reset_index(drop=True)
    return ctrl, supports_x, Ltot


def compute_auto_slopes(ctrl):
    """Fill NaN slopes using finite differences; keep forced slopes intact."""
    ctrl = ctrl.copy()
    xs = ctrl["x"].to_numpy(float)
    zs = ctrl["z"].to_numpy(float)
    ms = ctrl["slope"].to_numpy(float)

    for i in range(len(ctrl)):
        if np.isfinite(ms[i]):
            continue
        if i == 0:
            ms[i] = (zs[i+1] - zs[i]) / (xs[i+1] - xs[i])
        elif i == len(ctrl) - 1:
            ms[i] = (zs[i] - zs[i-1]) / (xs[i] - xs[i-1])
        else:
            ms[i] = (zs[i+1] - zs[i-1]) / (xs[i+1] - xs[i-1])

    ctrl["slope"] = ms
    ctrl.loc[ctrl["slope_mode"] != "forced", "slope_mode"] = "auto"
    return ctrl


def build_profile(ctrl, step):
    """Piecewise Hermite between consecutive control points."""
    ctrl = ctrl.sort_values("x").reset_index(drop=True)
    xs_all, zs_all = [], []

    for i in range(len(ctrl) - 1):
        x0, z0, m0 = float(ctrl.loc[i, "x"]), float(ctrl.loc[i, "z"]), float(ctrl.loc[i, "slope"])
        x1, z1, m1 = float(ctrl.loc[i+1, "x"]), float(ctrl.loc[i+1, "z"]), float(ctrl.loc[i+1, "slope"])

        if i == 0:
            xs = np.arange(x0, x1 + 1e-9, step)
        else:
            xs = np.arange(x0 + step, x1 + 1e-9, step)

        if len(xs) == 0:
            continue

        zs = hermite_eval(x0, z0, m0, x1, z1, m1, xs)
        xs_all.append(xs)
        zs_all.append(zs)

    x = np.concatenate(xs_all) if xs_all else np.array([0.0])
    z = np.concatenate(zs_all) if zs_all else np.array([0.0])
    return x, z


def enforce_end_linear_with_tangent(ctrl, supports_x, Le, step):
    """
    Enforce linear end segments of length Le with tangent match:
      - Determine z at x=Le and x=L-Le from preliminary curve
      - Force slopes at ends and at those points so the end segments are straight (C1).
    """
    Ltot = float(supports_x[-1])
    if Le <= 0 or Le >= Ltot / 2:
        return ctrl, None

    ctrl_tmp = compute_auto_slopes(ctrl)
    x0, z0 = build_profile(ctrl_tmp, step)

    z_at_Le = float(np.interp(Le, x0, z0))
    z_at_R  = float(np.interp(Ltot - Le, x0, z0))

    z_left  = float(ctrl.loc[(ctrl["x"] - 0.0).abs().idxmin(), "z"])
    z_right = float(ctrl.loc[(ctrl["x"] - Ltot).abs().idxmin(), "z"])

    mL = (z_at_Le - z_left) / Le
    mR = (z_right - z_at_R) / Le

    ctrl2 = ctrl.copy()
    ctrl2 = upsert_point(ctrl2, "END", Le, z_at_Le, mL)
    ctrl2 = upsert_point(ctrl2, "END", Ltot - Le, z_at_R, mR)
    ctrl2 = upsert_point(ctrl2, "SUP", 0.0, z_left, mL)
    ctrl2 = upsert_point(ctrl2, "SUP", Ltot, z_right, mR)

    return ctrl2.sort_values("x").reset_index(drop=True), {"Le": Le, "mL": mL, "mR": mR}


def check_inside_deck(z, t):
    zmin = float(np.min(z))
    zmax = float(np.max(z))
    ok = (zmin >= -1e-6) and (zmax <= t + 1e-6)
    return ok, zmin, zmax


# -----------------------------
# PDF Report Utilities
# -----------------------------
def fig_to_png_bytes(fig, dpi=200):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def df_to_table_data(df, max_rows=60):
    df2 = df.copy()
    if len(df2) > max_rows:
        df2 = df2.head(max_rows)
    data = [list(df2.columns)]
    for _, row in df2.iterrows():
        data.append([str(v) for v in row.values])
    return data


def fit_image_to_box(img_flowable, max_w, max_h):
    """Scale ReportLab Image flowable to fit inside (max_w, max_h) while preserving aspect ratio."""
    iw = float(img_flowable.imageWidth)
    ih = float(img_flowable.imageHeight)
    if iw <= 0 or ih <= 0:
        return img_flowable
    scale = min(max_w / iw, max_h / ih, 1.0)  # never upscale
    img_flowable.drawWidth = iw * scale
    img_flowable.drawHeight = ih * scale
    return img_flowable


def draw_header_footer(c: rl_canvas.Canvas, doc, meta, logo_bytes=None):
    """Header + footer with optional logo and page number."""
    page_w, page_h = A4
    left = doc.leftMargin
    right = page_w - doc.rightMargin
    bottom = doc.bottomMargin

    # Header line
    c.setStrokeColor(colors.grey)
    c.setLineWidth(0.5)
    c.line(left, page_h - 22*mm, right, page_h - 22*mm)

    # Logo
    if logo_bytes:
        try:
            logo = ImageReader(BytesIO(logo_bytes))
            lw, lh = 28*mm, 12*mm
            c.drawImage(
                logo,
                left, page_h - 20*mm - lh + 2*mm,
                width=lw, height=lh,
                preserveAspectRatio=True,
                mask='auto'
            )
        except Exception:
            pass

    # Title / info
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(left + 32*mm, page_h - 16*mm, meta.get("title", "PT Tendon Report (2D)"))

    c.setFont("Helvetica", 8)
    c.drawString(left + 32*mm, page_h - 20*mm, f"Generated: {meta.get('generated_at','')}")
    if meta.get("project"):
        c.drawRightString(right, page_h - 16*mm, f"Project: {meta.get('project')}")

    # Footer line
    c.setStrokeColor(colors.grey)
    c.setLineWidth(0.5)
    c.line(left, bottom + 10*mm, right, bottom + 10*mm)

    c.setFont("Helvetica", 8)
    c.setFillColor(colors.grey)
    c.drawString(left, bottom + 3*mm, meta.get("disclaimer_short", "For information only—verify independently."))
    c.drawRightString(right, bottom + 3*mm, f"Page {c.getPageNumber()}")


def build_pdf_report(payload, logo_bytes=None):
    """
    Professional PDF with header/footer, inputs, results, tables, and figures.
    Solves LayoutError by scaling images to fit.
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=15*mm,
        rightMargin=15*mm,
        topMargin=28*mm,     # room for header
        bottomMargin=18*mm,  # room for footer
        title=payload["meta"].get("title", "PT Tendon Report"),
        author=payload["meta"].get("author", "")
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=8, leading=10))
    styles.add(ParagraphStyle(name="Tiny", parent=styles["Normal"], fontSize=7, leading=9))
    styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], spaceAfter=6))
    styles.add(ParagraphStyle(name="H3", parent=styles["Heading3"], spaceAfter=4))

    meta = payload["meta"].copy()
    meta.setdefault("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    meta.setdefault("disclaimer_short", "Engineering tool output—verify independently.")
    meta.setdefault(
        "disclaimer_long",
        "DISCLAIMER: This report is generated by a preliminary calculation tool. "
        "Results depend on user inputs and modelling assumptions. "
        "It is the user's responsibility to verify and validate all results."
    )

    story = []
    story.append(Paragraph(meta.get("title", "PT Tendon Friction + Elongation Report (2D)"), styles["Title"]))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(meta["disclaimer_long"], styles["Small"]))
    story.append(Spacer(1, 6*mm))

    # Inputs (params)
    story.append(Paragraph("Inputs", styles["H2"]))
    params = payload["params"]
    params_rows = [[Paragraph("<b>Item</b>", styles["Small"]), Paragraph("<b>Value</b>", styles["Small"])]]
    for k, v in params.items():
        params_rows.append([Paragraph(str(k), styles["Small"]), Paragraph(str(v), styles["Small"])])

    t_params = Table(params_rows, colWidths=[60*mm, 110*mm], repeatRows=1)
    t_params.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_params)
    story.append(Spacer(1, 6*mm))

    def add_df_section(name, df, max_rows=60):
        story.append(Paragraph(name, styles["H3"]))
        data = df_to_table_data(df, max_rows=max_rows)

        pdata = []
        for r, row in enumerate(data):
            prow = []
            for cell in row:
                if r == 0:
                    prow.append(Paragraph(f"<b>{cell}</b>", styles["Tiny"]))
                else:
                    prow.append(Paragraph(str(cell), styles["Tiny"]))
            pdata.append(prow)

        table = Table(pdata, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(table)
        story.append(Spacer(1, 5*mm))

    add_df_section("Span table", payload["span_df"], max_rows=80)
    add_df_section("Support table", payload["support_df"], max_rows=80)
    add_df_section("Pier inflection table", payload["pier_df"], max_rows=80)
    add_df_section("Control points", payload["ctrl_df"], max_rows=120)

    story.append(PageBreak())

    # Results
    story.append(Paragraph("Results", styles["H2"]))
    res = payload["results"]
    res_rows = [[Paragraph("<b>Item</b>", styles["Small"]), Paragraph("<b>Value</b>", styles["Small"])]]
    for k, v in res.items():
        res_rows.append([Paragraph(str(k), styles["Small"]), Paragraph(str(v), styles["Small"])])

    t_res = Table(res_rows, colWidths=[70*mm, 100*mm], repeatRows=1)
    t_res.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_res)
    story.append(Spacer(1, 6*mm))

    # Figures (safe fit) — put each figure in its own KeepTogether block; also keep heights conservative
    story.append(Paragraph("Figures", styles["H2"]))

    avail_w = doc.width
    max_h = 90 * mm  # conservative to avoid overflow

    img1 = Image(BytesIO(payload["fig_geom_png"]))
    img1 = fit_image_to_box(img1, avail_w, max_h)

    img2 = Image(BytesIO(payload["fig_force_png"]))
    img2 = fit_image_to_box(img2, avail_w, max_h)

    story.append(KeepTogether([Paragraph("Tendon geometry (z vs x)", styles["H3"]), img1, Spacer(1, 5*mm)]))
    story.append(KeepTogether([Paragraph("Tendon force after friction (P vs x)", styles["H3"]), img2]))

    def on_first_page(c, d):
        draw_header_footer(c, d, meta, logo_bytes=logo_bytes)

    def on_later_pages(c, d):
        draw_header_footer(c, d, meta, logo_bytes=logo_bytes)

    doc.build(story, onFirstPage=on_first_page, onLaterPages=on_later_pages)

    buf.seek(0)
    return buf.getvalue()


# -----------------------------
# App UI
# -----------------------------
st.title("🏗️ Post-Tensioning Tendon Analysis System")
st.markdown("### Professional 2D PT Tendon Analysis with Friction & Elongation Calculation")

st.info(
    "📐 **Coordinate System:** All vertical inputs represent heights from deck soffit (bottom):  \n"
    "• z = 0 at soffit reference  \n"
    "• z = t at deck top surface  \n\n"
    "⚙️ **Boundary Conditions:** Interior supports enforce horizontal tangent (slope = 0)  \n"
    "🔧 **End Zones:** Optional straight end segments with tangent continuity matching"
)


with st.sidebar:
    st.header("📊 Geometric Configuration")
    
    st.subheader("Bridge Parameters")
    n_spans = st.number_input(
        "Number of Spans", 
        min_value=1, 
        max_value=30, 
        value=3, 
        step=1,
        help="Define the number of continuous spans"
    )
    deck_t = st.number_input(
        "Deck Thickness t (m)", 
        min_value=0.10, 
        max_value=10.00, 
        value=2.00, 
        step=0.01,
        help="Total structural depth from soffit to top surface"
    )

    st.subheader("Analysis Discretization")
    step = st.number_input(
        "Station Interval (m)", 
        min_value=0.02, 
        max_value=2.00, 
        value=0.25, 
        step=0.01,
        help="Spacing between calculation points along tendon"
    )

    st.subheader("Geometric Controls")
    min_gap = st.number_input(
        "Minimum Control Point Spacing (m)", 
        min_value=0.05, 
        max_value=5.0, 
        value=0.25, 
        step=0.05,
        help="Minimum allowable distance between adjacent control points"
    )

    st.subheader("Profile Constraints")
    zero_slope_low = st.checkbox(
        "Enforce Horizontal Tangent at Low Points", 
        value=True,
        help="Force zero slope at span low points for smoother profile"
    )
    end_linear = st.checkbox(
        "Apply Linear End Zones", 
        value=True,
        help="Create straight tendon segments at member ends with tangent matching"
    )
    Le = st.number_input(
        "End Zone Length Le (m)", 
        min_value=0.0, 
        max_value=50.0, 
        value=1.5, 
        step=0.1,
        help="Length of linear end zone from each support"
    )

    st.divider()
    
    st.header("⚡ Prestressing Parameters")
    
    st.subheader("Initial Force")
    P0 = st.number_input(
        "Jacking Force P₀ (kN)", 
        min_value=1.0, 
        max_value=1e6, 
        value=2000.0, 
        step=50.0,
        help="Prestressing force applied at jacking end"
    )
    
    st.subheader("Friction Coefficients")
    mu = st.number_input(
        "Curvature Friction μ (-)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.25, 
        step=0.01,
        help="Friction coefficient for angular displacement (typical: 0.15-0.30)"
    )
    k = st.number_input(
        "Wobble Coefficient k (1/m)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.0033, 
        step=0.0001, 
        format="%.6f",
        help="Friction coefficient for length-dependent wobble (typical: 0.0010-0.0050)"
    )

    st.subheader("Stressing Configuration")
    jacking_mode = st.selectbox(
        "Jacking Method", 
        ["Single-end (Left)", "Single-end (Right)", "Double-end"],
        help="Select the stressing sequence and configuration"
    )

    st.divider()
    
    st.header("🔬 Material Properties")
    
    st.subheader("Tendon Characteristics")
    A_mm2 = st.number_input(
        "Cross-sectional Area A (mm²)", 
        min_value=1.0, 
        max_value=1e6, 
        value=1400.0, 
        step=10.0,
        help="Total area of prestressing steel"
    )
    E_MPa = st.number_input(
        "Elastic Modulus E (MPa)", 
        min_value=10000.0, 
        max_value=300000.0, 
        value=200000.0, 
        step=1000.0,
        help="Modulus of elasticity of prestressing steel (typical: 195,000-205,000 MPa)"
    )

    st.divider()
    
    st.header("📄 Report Generation")
    
    project_name = st.text_input(
        "Project Name (optional)", 
        value="",
        help="Enter project identification name for the report"
    )
    author_name = st.text_input(
        "Engineer Name (optional)", 
        value="",
        help="Enter the responsible engineer's name"
    )
    logo_file = st.file_uploader(
        "Upload Logo for Report Header", 
        type=["png", "jpg", "jpeg"],
        help="Upload a company/project logo (PNG or JPG format)"
    )



# -----------------------------
# Default tables
# -----------------------------
st.markdown("## 📏 Step 1: Span Configuration & Low Point Definition")
st.caption("Define span lengths and position of lowest tendon points within each span")

default_L = [30.0] * int(n_spans)

# Default alpha:
# - if single span => 0.5
# - else first=0.4, interior=0.5, last=0.6 (=> 0.4 from right support)
default_alpha = []
if int(n_spans) == 1:
    default_alpha = [0.5]
else:
    for i in range(int(n_spans)):
        if i == 0:
            default_alpha.append(0.4)
        elif i == int(n_spans) - 1:
            default_alpha.append(0.6)
        else:
            default_alpha.append(0.5)

span_df = pd.DataFrame({
    "L (m)": default_L,
    "alpha (low point as fraction of L)": default_alpha,
    "low_h_from_soffit (m)": [0.30] * int(n_spans),
})
span_df = st.data_editor(span_df, num_rows="fixed", use_container_width=True, key="span_df_final_app")

st.markdown("## 🔺 Step 2: Support Elevation Definition")
st.caption("Specify tendon elevation at each support location (measured from soffit)")

support_df = pd.DataFrame({
    "support_id": list(range(1, int(n_spans) + 2)),
    "support_h_from_soffit (m)": [1.00] * (int(n_spans) + 1),
})
support_df = st.data_editor(support_df, num_rows="fixed", use_container_width=True, key="support_df_final_app")

st.markdown("## 📍 Step 3: Interior Pier Inflection Point Configuration")
st.caption("Define distances from pier centerline to inflection points on either side for continuous spans")

pier_rows = []
for sidx in range(int(n_spans) + 1):
    if sidx == 0 or sidx == int(n_spans):
        pier_rows.append({"support_index": sidx, "d_before (m)": 0.0, "d_after (m)": 0.0})
    else:
        pier_rows.append({"support_index": sidx, "d_before (m)": 5.0, "d_after (m)": 5.0})
pier_df = pd.DataFrame(pier_rows).set_index("support_index")
pier_df = st.data_editor(pier_df, num_rows="fixed", use_container_width=True, key="pier_df_final_app")

# -----------------------------
# Build profile
# -----------------------------
try:
    support_h = support_df["support_h_from_soffit (m)"].astype(float).to_numpy()
    ctrl, supports_x, Ltot = build_base_control_points(
        span_df=span_df,
        support_h=support_h,
        pier_df=pier_df,
        zero_slope_low=bool(zero_slope_low),
        min_gap=float(min_gap)
    )

    end_info = None
    if end_linear and float(Le) > 0:
        ctrl, end_info = enforce_end_linear_with_tangent(ctrl, supports_x, float(Le), float(step))

    ctrl = compute_auto_slopes(ctrl)
    x, z = build_profile(ctrl, step=float(step))
except Exception as e:
    st.error(f"Could not build tendon profile. Details: {e}")
    st.stop()

ok, zmin, zmax = check_inside_deck(z, deck_t)
if not ok:
    st.warning(
        f"Tendon goes outside deck thickness (0 ≤ z ≤ t={deck_t:.3f}). "
        f"Computed z range: [{zmin:.3f}, {zmax:.3f}] m. Adjust heights."
    )

# -----------------------------
# Friction forces
# -----------------------------
theta_left = compute_theta_from_profile(x, z)
theta_total = float(theta_left[-1])
Ltendon = true_tendon_length(x, z)

# Left jacking
P_left = friction_force(P0, mu, k, x, theta_left)
P_left = np.minimum(P_left, P0)  # safety clamp

# Right jacking: reparameterize x from right end
x_rev = (Ltot - x)  # right end becomes 0
order = np.argsort(x_rev)
x_r = x_rev[order]      # increasing 0..L
z_r = z[order]
theta_r = compute_theta_from_profile(x_r, z_r)
P_r_along = friction_force(P0, mu, k, x_r, theta_r)
P_r_along = np.minimum(P_r_along, P0)

P_right = np.empty_like(P_r_along)
P_right[order] = P_r_along
P_right[-1] = P0  # enforce exact at right end

# Double-end envelope
P_double = np.maximum(P_left, P_right)
P_double = np.minimum(P_double, P0)

if jacking_mode == "Single-end (Left)":
    P_selected = P_left
elif jacking_mode == "Single-end (Right)":
    P_selected = P_right
else:
    P_selected = P_double

imin = int(np.argmin(P_selected))
x_minP = float(x[imin])
P_min = float(P_selected[imin])

# -----------------------------
# Elongations
# -----------------------------
elong_left_mm = theoretical_elongation_mm(P_left, x, A_mm2, E_MPa)
elong_right_mm = theoretical_elongation_mm(P_right, x, A_mm2, E_MPa)
elong_double_mm = theoretical_elongation_mm(P_double, x, A_mm2, E_MPa)
elong_selected_mm = theoretical_elongation_mm(P_selected, x, A_mm2, E_MPa)

# -----------------------------
# Plots
# -----------------------------
st.markdown("---")
st.markdown("## 📊 Analysis Results Visualization")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 📐 Tendon Geometric Profile")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot([0, Ltot], [0, 0], 'k-', linewidth=1.5, label="Soffit Reference (z=0)", alpha=0.7)
    ax1.plot([0, Ltot], [deck_t, deck_t], 'k--', linewidth=1.5, label=f"Deck Top (z={deck_t:.2f}m)", alpha=0.7)
    ax1.plot(x, z, 'b-', linewidth=2.5, label="PT Tendon Profile (C¹ Hermite)")
    
    for xx in supports_x:
        ax1.axvline(xx, color='gray', linewidth=0.8, linestyle=':', alpha=0.6)
    
    if end_info is not None:
        ax1.axvline(end_info["Le"], linestyle='--', color='red', linewidth=1.2, alpha=0.7, label=f"End Zone Boundary (Le={end_info['Le']:.2f}m)")
        ax1.axvline(Ltot - end_info["Le"], linestyle='--', color='red', linewidth=1.2, alpha=0.7)
    
    ax1.scatter(ctrl["x"], ctrl["z"], s=50, c='red', zorder=5, edgecolors='black', linewidths=0.5, label="Control Points")
    
    ax1.set_xlabel("Longitudinal Position x (m)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Elevation z (m from soffit)", fontsize=11, fontweight='bold')
    ax1.set_title("Tendon Geometric Profile", fontsize=13, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.legend(loc="best", framealpha=0.9, fontsize=9)
    ax1.set_xlim(-1, Ltot + 1)
    
    st.pyplot(fig1, clear_figure=False)

with col2:
    st.markdown("### 📊 Prestressing Force Distribution")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    lw_base = 2
    lw_sel = 3.5
    lwL = lw_sel if jacking_mode == "Single-end (Left)" else lw_base
    lwR = lw_sel if jacking_mode == "Single-end (Right)" else lw_base
    lwD = lw_sel if jacking_mode == "Double-end" else lw_base
    
    ax2.plot(x, P_left, linewidth=lwL, label="Single-End Left", color='#1f77b4', alpha=0.8)
    ax2.plot(x, P_right, linewidth=lwR, linestyle="--", label="Single-End Right", color='#ff7f0e', alpha=0.8)
    ax2.plot(x, P_double, linewidth=lwD, label="Double-End Envelope", color='#2ca02c', alpha=0.8)
    
    ax2.scatter([x_minP], [P_min], s=120, c='red', zorder=5, edgecolors='black', linewidths=1.5, 
               label=f"Min Force = {P_min:.1f} kN @ x={x_minP:.2f}m", marker='D')
    
    ax2.axhline(P0, color='gray', linestyle=':', linewidth=1.5, alpha=0.6, label=f"Initial Force P₀={P0:.0f} kN")
    
    ax2.set_xlabel("Longitudinal Position x (m)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Prestressing Force P (kN)", fontsize=11, fontweight='bold')
    ax2.set_title("Force Distribution After Friction Losses", fontsize=13, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.legend(loc="best", framealpha=0.9, fontsize=9)
    ax2.set_xlim(-1, Ltot + 1)
    
    st.pyplot(fig2, clear_figure=False)

# -----------------------------
# Summary
# -----------------------------
st.markdown("---")
st.markdown("### 📈 Key Analysis Metrics")

s1, s2, s3, s4 = st.columns(4)
s1.metric(
    "Horizontal Span Length", 
    f"{Ltot:.3f} m",
    help="Total horizontal projection of the tendon"
)
s2.metric(
    "True Tendon Length", 
    f"{Ltendon:.3f} m",
    delta=f"+{(Ltendon - Ltot):.3f} m",
    help="Actual length along the curved tendon path"
)
s3.metric(
    "Total Angular Change", 
    f"{theta_total:.5f} rad",
    delta=f"{np.degrees(theta_total):.3f}°",
    help="Cumulative absolute change in tendon angle"
)
s4.metric(
    "Minimum Force (Selected)", 
    f"{P_min:.1f} kN",
    delta=f"-{P0 - P_min:.1f} kN",
    delta_color="inverse",
    help=f"Minimum force occurs at x = {x_minP:.2f} m"
)

st.markdown("### 📏 Theoretical Elongation Results")
e1, e2, e3, e4 = st.columns(4)
e1.metric(
    "Left-End Jacking", 
    f"{elong_left_mm:.3f} mm",
    help="Theoretical elongation for left-end stressing only"
)
e2.metric(
    "Right-End Jacking", 
    f"{elong_right_mm:.3f} mm",
    help="Theoretical elongation for right-end stressing only"
)
e3.metric(
    "Double-End Jacking", 
    f"{elong_double_mm:.3f} mm",
    help="Theoretical elongation for simultaneous stressing"
)
e4.metric(
    f"Selected Mode", 
    f"{elong_selected_mm:.3f} mm",
    help=f"Elongation for {jacking_mode}"
)

if end_info is not None:
    st.info(f"ℹ️ **End Zone Constraint Applied:** Le = {end_info['Le']:.3f} m | Left Slope = {end_info['mL']:.6f} | Right Slope = {end_info['mR']:.6f}")

# -----------------------------
# Control points & CSV
# -----------------------------
st.markdown("---")
st.markdown("### 🎯 Control Points Definition")
st.caption("Geometric control points defining the smooth tendon profile with C¹ continuity")

ctrl_display = ctrl.copy()
ctrl_display = ctrl_display.rename(columns={
    'type': 'Point Type',
    'x': 'Position x (m)',
    'z': 'Height z (m)',
    'slope': 'Tangent Slope',
    'slope_mode': 'Slope Mode'
})
st.dataframe(ctrl_display, use_container_width=True, hide_index=True)

st.markdown("### 💾 Export Analysis Data")

col_exp1, col_exp2 = st.columns([2, 1])

with col_exp1:
    st.markdown("**Station-by-Station Results**")
    dx = np.diff(x)
    dz = np.diff(z)
    ds = np.sqrt(dx**2 + dz**2)
    
    out_df = pd.DataFrame({
        "Station_x_m": x,
        "Elevation_z_m": z,
        "Cumulative_Angle_rad": theta_left,
        "Force_Left_kN": P_left,
        "Force_Right_kN": P_right,
        "Force_Double_kN": P_double,
        "Force_Selected_kN": P_selected,
    })
    out_df["Segment_Length_m"] = np.concatenate([[0.0], ds])
    out_df["Total_Curvature_rad"] = theta_total
    out_df["Total_Tendon_Length_m"] = Ltendon
    
    st.caption("Preview (first 20 rows)")
    st.dataframe(out_df.head(20), use_container_width=True, hide_index=True)

with col_exp2:
    st.markdown("**Download Options**")
    st.download_button(
        label="📥 Download Complete Dataset (CSV)",
        data=out_df.to_csv(index=False).encode("utf-8"),
        file_name=f"PT_Tendon_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )


# -----------------------------
# PDF report
# -----------------------------
st.markdown("---")
st.markdown("### 📄 Professional PDF Report Generation")

col_pdf1, col_pdf2 = st.columns([2, 1])

with col_pdf1:
    st.info(
        "📋 **Report Contents:**  \n"
        "• Complete input parameters and configuration  \n"
        "• Detailed analysis results and metrics  \n"
        "• Geometric profile and force distribution charts  \n"
        "• Control point definitions and support data  \n"
        "• Professional formatting with header/footer"
    )

with col_pdf2:
    # store logo bytes if uploaded
    logo_bytes = None
    if logo_file is not None:
        try:
            logo_bytes = logo_file.read()
        except Exception:
            logo_bytes = None

if st.button("🔨 Generate Professional PDF Report", use_container_width=True, type="primary"):
    with st.spinner("⏳ Generating comprehensive PDF report..."):
        fig_geom_png = fig_to_png_bytes(fig1)
        fig_force_png = fig_to_png_bytes(fig2)
        
        params = {
            "Number of Spans": int(n_spans),
            "Deck Thickness (m)": float(deck_t),
            "Station Interval (m)": float(step),
            "Minimum Control Point Spacing (m)": float(min_gap),
            "Zero Slope at Low Points": bool(zero_slope_low),
            "End Linear Zones Applied": bool(end_linear),
            "End Zone Length Le (m)": float(Le),
            "Initial Jacking Force P₀ (kN)": float(P0),
            "Curvature Friction μ": float(mu),
            "Wobble Coefficient k (1/m)": float(k),
            "Jacking Configuration": str(jacking_mode),
            "Tendon Area A (mm²)": float(A_mm2),
            "Elastic Modulus E (MPa)": float(E_MPa),
        }
        
        results = {
            "Horizontal Span Length (m)": f"{Ltot:.3f}",
            "True Tendon Length (m)": f"{Ltendon:.3f}",
            "Total Angular Change (rad)": f"{theta_total:.5f}",
            "Total Angular Change (degrees)": f"{np.degrees(theta_total):.3f}",
            "Minimum Force - Selected (kN)": f"{P_min:.1f}",
            "Minimum Force Location x (m)": f"{x_minP:.2f}",
            "Elongation - Left Jack (mm)": f"{elong_left_mm:.3f}",
            "Elongation - Right Jack (mm)": f"{elong_right_mm:.3f}",
            "Elongation - Double-End (mm)": f"{elong_double_mm:.3f}",
            "Elongation - Selected (mm)": f"{elong_selected_mm:.3f}",
        }
        
        meta = {
            "title": "Post-Tensioning Tendon Analysis Report",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "project": project_name.strip() if project_name else "Unnamed Project",
            "author": author_name.strip() if author_name else "Not Specified",
            "disclaimer_short": "Engineering Analysis Tool - Verify All Results",
            "disclaimer_long": (
                "PROFESSIONAL DISCLAIMER: This report is generated by an engineering analysis tool "
                "for preliminary design purposes. All results are based on user-defined inputs and "
                "underlying analytical assumptions. The user bears full responsibility for verification "
                "and validation of all results. This tool does not replace professional engineering judgment "
                "or detailed design calculations. Always verify results against applicable design codes and standards."
            ),
        }
        
        payload = {
            "meta": meta,
            "params": params,
            "span_df": span_df.reset_index(drop=True),
            "support_df": support_df.reset_index(drop=True),
            "pier_df": pier_df.reset_index(),
            "ctrl_df": ctrl_display.reset_index(drop=True),
            "results": results,
            "fig_geom_png": fig_geom_png,
            "fig_force_png": fig_force_png,
        }
        
        try:
            pdf_bytes = build_pdf_report(payload, logo_bytes=logo_bytes)
            
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"PT_Tendon_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            st.success("✅ PDF report generated successfully!")
            
        except Exception as e:
            st.error(f"❌ PDF generation failed: {str(e)}")
            st.warning("Please check your inputs and try again. If the problem persists, contact support.")


st.markdown("---")

# Footer with technical notes
with st.expander("📚 Technical Notes & Methodology", expanded=False):
    st.markdown("""
    ### Analysis Methodology
    
    **Geometric Profile Definition:**
    - Cubic Hermite spline interpolation (C¹ continuity) ensures smooth tendon profiles
    - Control points defined at supports, low points, and inflection zones
    - Optional enforcement of zero slopes at span low points for improved geometry
    - Linear end zones can be specified to match anchorage details
    
    **Default Low Point Positioning:**
    - Single span: α = 0.5 (midspan position)
    - Multiple spans: First span α = 0.4, interior spans α = 0.5, last span α = 0.6
    - User can override these values for specific design requirements
    
    **Friction Analysis:**
    - Based on exponential friction model: P(x) = P₀ · exp(-(μθ + kx))
    - μ: Curvature friction coefficient (typical range: 0.15-0.25 for post-tensioning)
    - k: Wobble coefficient (typical range: 0.0010-0.0020 per meter)
    - Double-end analysis uses envelope of max(P_left, P_right) with upper bound P₀
    
    **Elongation Calculation:**
    - Theoretical elongation: Δ = ∫[P(x)dx] / (A·E)
    - Numerical integration via trapezoidal rule
    - Results represent elastic elongation under initial prestress conditions
    
    ### Important Considerations
    
    ⚠️ **Design Verification Required:**
    - All results must be independently verified by a licensed professional engineer
    - Code compliance checks for minimum cover, spacing, and other requirements
    - Material properties should be confirmed with supplier specifications
    - Local design codes and standards must be consulted
    
    ⚠️ **Limitations:**
    - Analysis assumes linear elastic behavior
    - Time-dependent losses (creep, shrinkage, relaxation) not included
    - Anchorage seating losses require separate calculation
    - Secondary effects and 3D geometry not modeled
    
    ### References
    - AASHTO LRFD Bridge Design Specifications
    - ACI 318: Building Code Requirements for Structural Concrete
    - PTI: Post-Tensioning Manual (6th Edition)
    """)

st.markdown("---")
st.caption("Post-Tensioning Analysis System v2.0 | Professional Engineering Tool | © 2026")

