#!/usr/bin/env python3
"""Regenerate the retained baseline figures for the IJAR manuscript.

The visual set is deliberately organized around the paper's reasoning chain:
workflow, exact three-state geometry, design capacity, sampling-unit validity,
the cross-benchmark operating point, multi-seed stability, breadth, and
lighting/domain-shift mechanisms. Quantitative panels use the frozen values
reported in the manuscript and are guarded by lightweight assertions below.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch, Polygon, Rectangle
import numpy as np
import csv
from pathlib import Path

plt.rcParams.update({
    # Match the manuscript's Times-like serif typography and embed vector text.
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 9.3,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.bbox": "tight",
})

NAVY = "#174A6E"
BLUE = "#2F78A8"
TEAL = "#2A9D8F"
GREEN = "#3A8D5D"
AMBER = "#E9A23B"
RED = "#C84A3A"
PURPLE = "#7B61A8"
GREY = "#D9DEE3"
INK = "#24323D"
FIGURE_DIR = Path(__file__).resolve().parent / "artifacts" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, stem: str, dpi: int = 600) -> None:
    fig.savefig(FIGURE_DIR / f"{stem}.pdf")
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=dpi)
    plt.close(fig)
    print(f"wrote {FIGURE_DIR / stem}.pdf/png")


def rounded_box(ax, xy, width, height, text, face, edge=NAVY, fontsize=8.4):
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=face, edgecolor=edge, linewidth=1.05,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2, xy[1] + height / 2, text,
        ha="center", va="center", fontsize=fontsize, color=INK,
        linespacing=1.18,
    )
    return box


# ---------------- Detector-agnostic AI validation architecture ----------------
fig0, ax0 = plt.subplots(figsize=(7.55, 3.45))
ax0.set_xlim(0, 1)
ax0.set_ylim(0, 1)
ax0.axis("off")

panel_x = [0.010, 0.260, 0.510, 0.760]
panel_w, panel_y, panel_h = 0.225, 0.105, 0.785
panel_faces = ["#F7FAFC", "#FFF9ED", "#FAF8FC", "#F7F8F8"]
header_colors = [NAVY, "#B87818", PURPLE, "#53636C"]
panel_titles = [
    "1   Locked AI pipeline",
    "2   Traceability constraints",
    "3   Validation layer",
    "4   Engineering action",
]
for xpos, face, header, title in zip(
        panel_x, panel_faces, header_colors, panel_titles):
    ax0.add_patch(Rectangle(
        (xpos, panel_y), panel_w, panel_h,
        facecolor=face, edgecolor="#9AA5AC", linewidth=0.9,
    ))
    ax0.add_patch(Rectangle(
        (xpos, 0.785), panel_w, 0.105,
        facecolor=header, edgecolor=header, linewidth=0.9,
    ))
    ax0.text(xpos + 0.012, 0.837, title, ha="left", va="center",
             fontsize=7.6, weight="bold", color="white")

for left, right in zip(panel_x[:-1], panel_x[1:]):
    ax0.add_patch(FancyArrowPatch(
        (left + panel_w - 0.010, 0.495), (right + 0.010, 0.495),
        arrowstyle="-|>", mutation_scale=11, linewidth=1.35, color=NAVY,
        shrinkA=0, shrinkB=0, zorder=8,
    ))

# Panel 1: a complete, immutable AI scorer produces score-bearing views.
rounded_box(
    ax0, (0.032, 0.716), 0.180, 0.040,
    r"LOCKED AI SCORER  $s(x)$", "#E8F1F7", BLUE, fontsize=6.6,
)
grid = np.linspace(-1.0, 1.0, 18)
gx, gy = np.meshgrid(grid, grid)
view_x = [0.050, 0.112, 0.174]
row_specs = [
    ("normal item", 0.515, [0.12, 0.18, 0.15], False),
    ("defect item", 0.305, [0.34, 0.78, 0.69], True),
]
for row_index, (row_label, ypos, scores, defective) in enumerate(row_specs):
    ax0.text(0.026, ypos + 0.155, row_label, ha="left", va="bottom",
             fontsize=7.2, weight="bold", color=INK)
    for view_index, (xpos, score) in enumerate(zip(view_x, scores)):
        base = 0.18 + 0.58 * np.exp(
            -((gx - 0.11 * (view_index - 1)) ** 2 / 0.34
              + (gy + 0.05 * (view_index - 1)) ** 2 / 0.20)
        )
        base += 0.08 * np.cos(3.2 * gx + 0.8 * view_index) * np.cos(2.4 * gy)
        if defective:
            hx = -0.42 + 0.40 * view_index
            hy = 0.30 - 0.22 * view_index
            base += 0.50 * np.exp(-((gx - hx) ** 2 + (gy - hy) ** 2) / 0.055)
        image_data = np.clip(base, 0, 1)
        ax0.imshow(
            image_data, extent=(xpos, xpos + 0.046, ypos, ypos + 0.125),
            origin="lower", cmap="bone_r", vmin=0, vmax=1,
            interpolation="bilinear", aspect="auto", zorder=2,
        )
        ax0.add_patch(Rectangle(
            (xpos, ypos), 0.046, 0.125, facecolor="none",
            edgecolor="#70808A", linewidth=0.65, zorder=3,
        ))
        if defective:
            # Map the synthetic hotspot centre through the same image extent.
            # This keeps every red box registered to the injected defect centre.
            hotspot_x = xpos + 0.046 * (hx + 1.0) / 2.0
            hotspot_y = ypos + 0.125 * (hy + 1.0) / 2.0
            ax0.add_patch(Rectangle(
                (hotspot_x - 0.007, hotspot_y - 0.012), 0.014, 0.024,
                facecolor="none", edgecolor=RED, linewidth=0.85, zorder=4,
            ))
        bar_color = RED if score >= 0.60 else BLUE
        ax0.add_patch(Rectangle(
            (xpos, ypos - 0.026), 0.046, 0.010,
            facecolor="#DDE3E7", edgecolor="none", zorder=2,
        ))
        ax0.add_patch(Rectangle(
            (xpos, ypos - 0.026), 0.046 * score, 0.010,
            facecolor=bar_color, edgecolor="none", zorder=3,
        ))
        ax0.text(xpos + 0.023, ypos - 0.043, rf"$s_{view_index + 1}$",
                 ha="center", va="top", fontsize=6.4, color="#59656E")
ax0.plot([0.034, 0.220], [0.190, 0.190], color=RED, lw=0.8, ls=(0, (2, 2)))
ax0.text(0.127, 0.168, r"locked threshold $\tau$; no retuning",
         ha="center", va="top", fontsize=6.5, color=RED)

# Panel 2: observed IDs versus a family of compatible partitions.
cell_x = [0.287 + 0.029 * index for index in range(6)]
cell_w = 0.024
ax0.text(0.276, 0.730, "IDs observed", ha="left", va="center",
         fontsize=7.2, weight="bold", color=GREEN)
for index, xpos in enumerate(cell_x):
    label = "A" if index < 3 else "B"
    color = GREEN if label == "A" else BLUE
    ax0.add_patch(Rectangle(
        (xpos, 0.625), cell_w, 0.070,
        facecolor="white", edgecolor=color, linewidth=0.8,
    ))
    ax0.text(xpos + cell_w / 2, 0.660, label, ha="center", va="center",
             fontsize=7.0, weight="bold", color=color)
ax0.add_patch(Rectangle((cell_x[0], 0.590), 3 * 0.029 - 0.005, 0.010,
                        facecolor=GREEN, edgecolor="none"))
ax0.add_patch(Rectangle((cell_x[3], 0.590), 3 * 0.029 - 0.005, 0.010,
                        facecolor=BLUE, edgecolor="none"))
ax0.text(0.335, 0.566, "item A", ha="center", va="top", fontsize=6.5, color=GREEN)
ax0.text(0.421, 0.566, "item B", ha="center", va="top", fontsize=6.5, color=BLUE)

ax0.plot([0.278, 0.472], [0.520, 0.520], color="#D0D6DA", lw=0.7)
ax0.text(0.276, 0.480, "IDs missing", ha="left", va="center",
         fontsize=7.2, weight="bold", color="#9A6512")
for xpos in cell_x:
    ax0.add_patch(Rectangle(
        (xpos, 0.395), cell_w, 0.060,
        facecolor="white", edgecolor=AMBER, linewidth=0.75,
    ))
    ax0.text(xpos + cell_w / 2, 0.425, "?", ha="center", va="center",
             fontsize=7.0, color="#9A6512")
partition_rows = [
    ("$P_1$", 0.335, [(0, 3, GREEN), (3, 6, BLUE)]),
    ("$P_2$", 0.275, [(0, 2, GREEN), (2, 5, PURPLE), (5, 6, BLUE)]),
]
for label, ypos, groups in partition_rows:
    ax0.text(0.280, ypos + 0.008, label, ha="left", va="center",
             fontsize=6.8, color=INK)
    for start, stop, color in groups:
        # Reserve a label gutter so P_1/P_2 never overlap the partition bars.
        x0 = 0.312 + 0.026 * start
        x1 = 0.312 + 0.026 * stop - 0.003
        ax0.add_patch(Rectangle(
            (x0, ypos), x1 - x0, 0.016,
            facecolor=color, edgecolor="white", linewidth=0.4, alpha=0.88,
        ))
ax0.add_patch(Rectangle(
    (0.300, 0.160), 0.155, 0.052,
    facecolor="#F8D99D", edgecolor=AMBER, linewidth=0.9,
))
ax0.text(0.3775, 0.186, r"metadata $(F,A,N,r)$",
         ha="center", va="center", fontsize=7.7, weight="bold", color="#744A0B")
ax0.text(0.3775, 0.134, "predeclared outer constraint set",
         ha="center", va="center", fontsize=6.2, color="#744A0B")

# Panel 3: derive the sharp count set, then exact component decisions.
ax0.add_patch(Rectangle(
    (0.545, 0.680), 0.155, 0.050,
    facecolor="#F8D99D", edgecolor=AMBER, linewidth=0.9,
))
ax0.text(0.6225, 0.701, r"$K\in[K_-,K_+]$",
         ha="center", va="center", fontsize=7.8, weight="bold", color="#744A0B")
ax0.text(0.622, 0.635, "exact one-sided item bounds",
         ha="center", va="center", fontsize=7.0, weight="bold", color=INK)
audit_rows = [
    ("false reject", "FR", 0.505, 0.566, 0.631, BLUE, r"$\alpha_0$"),
    ("missed defect", "MD", 0.335, 0.618, 0.694, PURPLE, r"$\alpha_1$"),
]
budget_x = 0.650
for long_label, short_label, ypos, left, right, color, budget_label in audit_rows:
    ax0.text(0.528, ypos + 0.055, long_label, ha="left", va="center",
             fontsize=6.8, color=INK)
    ax0.text(0.528, ypos, short_label, ha="left", va="center",
             fontsize=7.4, weight="bold", color=color)
    ax0.plot([0.560, 0.713], [ypos, ypos], color="#AAB4BA", lw=0.75)
    ax0.plot([budget_x, budget_x], [ypos - 0.048, ypos + 0.048],
             color=RED, lw=0.85, ls=(0, (2, 2)))
    ax0.text(budget_x, ypos + 0.058, budget_label, ha="center", va="bottom",
             fontsize=6.6, color=RED)
    ax0.plot([left, right], [ypos, ypos], color=color, lw=2.2,
             solid_capstyle="butt", zorder=3)
    ax0.plot([left, left], [ypos - 0.018, ypos + 0.018],
             color=color, lw=1.0, zorder=4)
    ax0.plot([right, right], [ypos - 0.018, ypos + 0.018],
             color=color, lw=1.0, zorder=4)
    ax0.scatter([(left + right) / 2], [ypos], s=13, facecolor="white",
                edgecolor=color, linewidth=0.8, zorder=5)
ax0.text(0.622, 0.235, "component decision", ha="center", va="center",
         fontsize=6.8, color="#59656E")
state_specs = [
    ("C", "certified", GREEN, "#E4F1E8"),
    ("U", "unresolved", "#7D878E", "#ECEFF1"),
    ("V", "violated", RED, "#F8E7E4"),
]
for index, (letter, label, edge, face) in enumerate(state_specs):
    xpos = 0.543 + 0.063 * index
    ax0.add_patch(Rectangle(
        (xpos, 0.145), 0.052, 0.058,
        facecolor=face, edgecolor=edge, linewidth=0.9,
    ))
    ax0.text(xpos + 0.026, 0.172, letter, ha="center", va="center",
             fontsize=8.0, weight="bold", color=edge)
    ax0.text(xpos + 0.026, 0.133, label, ha="center", va="top",
             fontsize=5.6, color=INK)

# Panel 4: the intersection-union rule as a compact, auditable decision matrix.
ax0.text(0.872, 0.733, "missed-defect state", ha="center", va="center",
         fontsize=7.0, weight="bold", color=INK)
matrix_x0, matrix_y0 = 0.817, 0.410
cell = 0.046
state_letters = ["C", "U", "V"]
for col, letter in enumerate(state_letters):
    ax0.text(matrix_x0 + (col + 0.5) * cell, matrix_y0 + 3 * cell + 0.026,
             letter, ha="center", va="center", fontsize=7.0,
             weight="bold", color=INK)
for row, letter in enumerate(state_letters):
    ypos = matrix_y0 + (2 - row + 0.5) * cell
    ax0.text(matrix_x0 - 0.014, ypos, letter, ha="center", va="center",
             fontsize=7.0, weight="bold", color=INK)
ax0.text(matrix_x0 - 0.035, matrix_y0 + 1.5 * cell, "FR",
         ha="center", va="center", fontsize=6.8, weight="bold",
         color=BLUE, rotation=90)
for row, fr_state in enumerate(state_letters):
    for col, md_state in enumerate(state_letters):
        if fr_state == "C" and md_state == "C":
            code, face, text_color = "R", GREEN, "white"
        elif fr_state == "V" or md_state == "V":
            code, face, text_color = "N", RED, "white"
        else:
            code, face, text_color = "T", "#C9D0D4", INK
        xpos = matrix_x0 + col * cell
        ypos = matrix_y0 + (2 - row) * cell
        ax0.add_patch(Rectangle(
            (xpos, ypos), cell, cell,
            facecolor=face, edgecolor="white", linewidth=0.8,
        ))
        ax0.text(xpos + cell / 2, ypos + cell / 2 - 0.002, code,
                 ha="center", va="center", fontsize=7.0,
                 weight="bold", color=text_color)
legend_rows = [
    ("R", "release", GREEN),
    ("T", "test more", "#7D878E"),
    ("N", "not release-ready", RED),
]
for index, (code, label, color) in enumerate(legend_rows):
    ypos = 0.325 - 0.070 * index
    ax0.add_patch(Rectangle(
        (0.800, ypos - 0.016), 0.030, 0.032,
        facecolor=(color if code != "T" else "#C9D0D4"),
        edgecolor=color, linewidth=0.7,
    ))
    ax0.text(0.815, ypos - 0.002, code, ha="center", va="center",
             fontsize=6.5, weight="bold",
             color=("white" if code != "T" else INK))
    ax0.text(0.840, ypos, label, ha="left", va="center",
             fontsize=6.8, color=INK)
ax0.text(0.872, 0.130, "release: C + C   |   change: revalidate",
         ha="center", va="center", fontsize=5.9, weight="bold", color=GREEN)

save(fig0, "fig0_validation_architecture")

# ----------------------------------------------------------------------------
# Fig 1 data: (dataset, n0, observed FPR, state) empirical PatchCore seed-0.
# AD2 empirical fp from the X2 paired table; MVTec/VisA from the P2 batch pull.
# AD2-raw (rule transport) fp shown as operating-point shift arrows.
AD2_UNITS = ["can","fabric","fruit_jelly","rice","sheet_metal","vial","wallplugs","walnuts"]
ad2_n0 = dict(zip(AD2_UNITS, [12,11,5,7,4,5,10,10]))
ad2_emp_fp = dict(zip(AD2_UNITS, [0,0,1,1,0,0,2,0]))      # empirical-quantile, seed-0
ad2_raw_fp = dict(zip(AD2_UNITS, [7,7,1,4,2,1,3,5]))      # raw mean+3sigma, seed-0
mvtec = [("screw",41,2),("carpet",28,10),("toothbrush",12,4),("capsule",23,1),
         ("pill",26,1),("tile",33,1),("hazelnut",40,0),("leather",32,9),
         ("zipper",32,4),("cable",58,3),("metal_nut",22,3),("bottle",20,0),
         ("wood",19,5),("transistor",60,9),("grid",21,1)]
visa = [("chewinggum",50,1),("macaroni1",100,3),("pcb4",101,7),("candle",100,4),
        ("capsules",60,3),("pipe_fryum",50,3),("cashew",50,1),("macaroni2",100,7),
        ("pcb2",100,9),("pcb3",101,5),("pcb1",100,9),("fryum",50,0)]

def fpr(fp,n0): return fp/n0 if n0 else 0.0

# CP one-sided lower bound to flag "violated" at alpha=0.05, delta=0.05
from scipy.stats import beta as sbeta
def violated(fp,n0,a=0.05,d=0.05):
    if n0==0: return False
    lo = float(sbeta.ppf(d, fp, n0-fp+1)) if fp>0 else 0.0
    return lo > a


def certified(fp, n0, a=0.05, d=0.05):
    if n0 == 0:
        return False
    hi = float(sbeta.ppf(1 - d, fp + 1, n0 - fp)) if fp < n0 else 1.0
    return hi <= a


# ---------------- Exact three-state decision geometry ----------------
ns = np.arange(2, 351)
cert_edge = np.zeros_like(ns, dtype=float)
viol_edge = np.ones_like(ns, dtype=float)
for idx, n in enumerate(ns):
    cert_ks = [k for k in range(n + 1) if certified(k, n)]
    viol_ks = [k for k in range(n + 1) if violated(k, n)]
    cert_edge[idx] = ((max(cert_ks) + 0.5) / n) if cert_ks else 0.0
    viol_edge[idx] = ((min(viol_ks) - 0.5) / n) if viol_ks else 1.0

fig3, ax3 = plt.subplots(figsize=(7.0, 3.8))
ax3.fill_between(ns, 0, cert_edge, color="#B9DFC5", alpha=0.95,
                 hatch="///", edgecolor="#647A69", linewidth=0.0,
                 label="certified")
ax3.fill_between(ns, cert_edge, viol_edge, color="#E2E6E9", alpha=0.95,
                 edgecolor="#90979C", linewidth=0.0, label="unresolved")
ax3.fill_between(ns, viol_edge, 0.42, color="#F2B8B2", alpha=0.95,
                 hatch="\\\\\\", edgecolor="#8D625E", linewidth=0.0,
                 label="violated")
ax3.plot(ns, cert_edge, color=GREEN, lw=1.2)
ax3.plot(ns, viol_edge, color=RED, lw=1.2, ls="--")
ax3.axhline(0.05, color=INK, lw=1.0, ls="--")
ax3.axvline(59, color=NAVY, lw=1.0, ls=":")
points = [(4, 0, "4 items, 0 FP", GREY, "s"),
          (59, 0, "59 items, 0 FP", GREEN, "o"),
          (60, 1, "60 items, 1 FP", AMBER, "^")]
for n, k, label, color, marker in points:
    ax3.scatter(n, k / n, s=45, color=color, marker=marker,
                edgecolor=INK, lw=0.7, zorder=5)
    ax3.annotate(label, (n, k / n), xytext=(5, 8), textcoords="offset points",
                 fontsize=8.3, color=INK)
ax3.set_xscale("log")
ax3.set_xlim(2, 350)
ax3.set_ylim(0, 0.42)
ax3.set_xlabel(r"independent normal items  $n_0$  (log scale)")
ax3.set_ylabel(r"observed false-positive fraction  $k/n_0$")
ax3.set_title(r"Exact one-sided three-state geometry  ($\alpha=\delta=5\%$)")
ax3.legend(handles=[Patch(facecolor="#B9DFC5", edgecolor="#647A69",
                          hatch="///", label="certified"),
                    Patch(facecolor="#E2E6E9", edgecolor="#90979C",
                          label="unresolved"),
                    Patch(facecolor="#F2B8B2", edgecolor="#8D625E",
                          hatch="\\\\\\", label="violated")],
           loc="upper right", frameon=False, ncol=3, fontsize=8.5)
ax3.text(320, 0.055, r"claim budget $\alpha$", ha="right", va="bottom",
         fontsize=8.4, color=INK)
fig3.tight_layout()
save(fig3, "fig3_three_state_geometry")


# ---------------- Zero-event capacity planner ----------------
alphas = np.linspace(0.005, 0.10, 400)
deltas = [(0.10, "90% confidence", TEAL, "--", "^"),
          (0.05, "95% confidence", NAVY, "-", "o"),
          (0.01, "99% confidence", PURPLE, "-.", "s")]
fig4, ax4 = plt.subplots(figsize=(6.6, 3.65))
for d, label, color, linestyle, marker in deltas:
    floors = np.ceil(np.log(d) / np.log(1 - alphas))
    ax4.plot(100 * alphas, floors, lw=1.8, label=label, color=color,
             ls=linestyle, marker=marker, markevery=70, ms=4.2,
             markerfacecolor="white", markeredgewidth=0.8)
ax4.scatter([1, 5], [299, 59], s=48, color=NAVY, edgecolor="white", zorder=5)
ax4.annotate("1% budget: 299 items", (1, 299), xytext=(12, 9),
             textcoords="offset points", fontsize=8.5, color=NAVY)
ax4.annotate("5% budget: 59 items", (5, 59), xytext=(9, 10),
             textcoords="offset points", fontsize=8.5, color=NAVY)
ax4.set_yscale("log")
ax4.set_xlim(0.5, 10)
ax4.set_ylim(20, 1000)
ax4.set_xlabel(r"pre-declared item-FPR budget  $\alpha$  (%)")
ax4.set_ylabel("minimum independent items with 0 FP")
ax4.set_title("Required independent items for zero-event certification")
ax4.grid(axis="y", which="both", color="#E7EAED", lw=0.7)
ax4.legend(loc="upper right", fontsize=8.5, frameon=False)
fig4.tight_layout()
save(fig4, "fig4_capacity_planner")


# ---------------- Sampling-unit correction ----------------
fig5, (ax5a, ax5b, ax5c) = plt.subplots(1, 3, figsize=(7.35, 2.85),
                                        gridspec_kw={"width_ratios": [1.12, 0.9, 1.0]})
for ax in (ax5a, ax5b, ax5c):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

ax5a.set_title("(a) Observed frames", color=NAVY, pad=7, fontsize=9.5)
rows = [(0.78, [1, 1, 0, 0, 0], "item A"),
        (0.55, [1, 0, 0, 0, 0], "item B"),
        (0.32, [1, 0, 0, 0, 0], "item C")]
for y, alarms, label in rows:
    ax5a.text(0.01, y, label, ha="left", va="center", fontsize=8.3, color=INK)
    for j, alarm in enumerate(alarms):
        ax5a.add_patch(FancyBboxPatch(
            (0.24 + j * 0.12, y - 0.048), 0.082, 0.096,
            boxstyle="round,pad=0.006", facecolor=(RED if alarm else "#E3E8EC"),
            edgecolor=(INK if alarm else "#FFFFFF"), linewidth=0.7,
            hatch=("///" if alarm else None),
        ))
ax5a.text(0.50, 0.13, r"observed: $A/F=4/55$", ha="center",
          fontsize=8.6, color=INK, weight="bold")
ax5a.text(0.50, 0.045, "frame IID analysis is inadmissible",
          ha="center", fontsize=8.2, color="#65727C")

ax5b.set_title("(b) IDs hidden", color=NAVY, pad=7, fontsize=9.5)
ax5b.plot([0.17, 0.83], [0.58, 0.58], color=TEAL, lw=2.0)
for value in range(1, 5):
    xpos = 0.17 + (value - 1) * 0.22
    ax5b.scatter(xpos, 0.58, s=52, color=TEAL, edgecolor="white", zorder=3)
    ax5b.text(xpos, 0.43, str(value), ha="center", va="center",
              fontsize=8.5, color=INK)
ax5b.text(0.50, 0.76, r"$K\in\{1,2,3,4\}$", ha="center",
          fontsize=9.5, color=TEAL, weight="bold")
ax5b.text(0.50, 0.28, r"given $N=11,\ r=5$", ha="center",
          fontsize=8.3, color="#65727C")
ax5b.text(0.50, 0.11, "partition-robust:\nunresolved", ha="center",
          va="center", fontsize=8.5, color=INK, weight="bold")

ax5c.set_title("(c) IDs verified", color=NAVY, pad=7, fontsize=9.5)
positions = [(0.18 + 0.16 * col, 0.76 - 0.18 * row)
             for row in range(3) for col in range(4)][:11]
for index, (xpos, ypos) in enumerate(positions):
    alarm = index < 3
    ax5c.scatter(xpos, ypos, s=78, color=(RED if alarm else "#DCE2E6"),
                 marker=("X" if alarm else "o"), edgecolor=INK,
                 linewidth=0.7)
ax5c.text(0.50, 0.25, r"mapped: $K/N=3/11$", ha="center",
          fontsize=8.6, color=INK, weight="bold")
ax5c.text(0.50, 0.11, "exact item audit:\nviolated", ha="center",
          va="center", fontsize=8.5, color=RED, weight="bold")

fig5.suptitle("Sampling-unit evidence under incomplete frame-to-item metadata", y=1.03,
               fontsize=10.5, weight="bold", color=INK)
fig5.tight_layout(w_pad=0.8)
save(fig5, "fig5_sampling_unit")

# ---------------- Fig 1 ----------------
fig, ax = plt.subplots(figsize=(7.15, 4.75))
# capacity-bound shading (n0 < 59 cannot certify at 5%, zero-event floor)
ax.axvspan(0, 59, facecolor="#E5E5E5", alpha=0.62, hatch="///",
           edgecolor="#A0A0A0", linewidth=0.0, zorder=0)
ax.axvline(59, color="#b00", ls="--", lw=1.3, zorder=1)
ax.axvline(299, color="#7a3a00", ls=":", lw=1.3, zorder=1)
ax.axhline(0.05, color="0.4", ls="-", lw=0.9, zorder=1)
COL = {"AD2":"#1f77b4","MVTec":"#2ca02c","VisA":"#9467bd"}
DATASET_MARKER = {"AD2": "o", "MVTec": "s", "VisA": "^"}
def scatter(ds, items, n0map=None, fpmap=None):
    for u,n0,fp in items:
        is_violated = violated(fp, n0)
        ax.scatter(n0, fpr(fp,n0), s=46, marker=DATASET_MARKER[ds],
                   facecolors=(COL[ds] if is_violated else "white"),
                   edgecolors=COL[ds], linewidths=1.0, zorder=3)
        if is_violated:
            ax.scatter(n0, fpr(fp,n0), s=24, marker="x", color=INK,
                       linewidths=0.8, zorder=4)
scatter("MVTec", mvtec)
scatter("VisA", visa)
# AD2 empirical
for u in AD2_UNITS:
    n0=ad2_n0[u]; fp=ad2_emp_fp[u]
    is_violated = violated(fp, n0)
    ax.scatter(n0, fpr(fp,n0), s=58, marker=DATASET_MARKER["AD2"],
               facecolors=(COL["AD2"] if is_violated else "white"),
               edgecolors=COL["AD2"], linewidths=1.1, zorder=4)
    if is_violated:
        ax.scatter(n0, fpr(fp,n0), s=28, marker="x", color=INK,
                   linewidths=0.8, zorder=5)
# AD2 empirical -> raw operating-point shift arrows (rule transport)
for u in AD2_UNITS:
    n0=ad2_n0[u]
    y0=fpr(ad2_emp_fp[u],n0); y1=fpr(ad2_raw_fp[u],n0)
    if abs(y1-y0) > 1e-9:
        ax.annotate("", xy=(n0,y1), xytext=(n0,y0),
                    arrowprops=dict(arrowstyle="->", color="#4A4A4A", lw=1.0,
                                    ls="--", alpha=0.85))
ax.set_xscale("log")
ax.set_xlim(3, 330); ax.set_ylim(0, 0.72)
ax.set_xlabel(r"independent normal items per category  $n_0$  (log scale)",
              labelpad=16)
ax.set_ylabel(r"observed item-decision FPR  ($\mathrm{fp}/n_0$)")
ax.text(0.30, 0.95, r"capacity-bound: $n_0<59$"+"\n"+"cannot certify at 5%",
        transform=ax.transAxes, fontsize=8.6, color="#7a1010",
        ha="center", va="top")
for x, label, color in [(59, "59", "#b00"), (299, "299", "#7a3a00")]:
    ax.annotate(label, xy=(x, 0), xycoords=("data", "axes fraction"),
                xytext=(0, -13), textcoords="offset points", ha="center",
                va="top", fontsize=8.5, color=color, annotation_clip=False)
ax.text(0.89, 0.085, r"$\alpha=5\%$ budget", transform=ax.transAxes,
        fontsize=8.5, color="0.3", ha="center",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.88))
legend_handles = [
    Line2D([0], [0], marker="o", markerfacecolor="white",
           markeredgecolor=COL["AD2"], lw=0, label="AD2 (empirical)"),
    Line2D([0], [0], marker="s", markerfacecolor="white",
           markeredgecolor=COL["MVTec"], lw=0, label="MVTec AD"),
    Line2D([0], [0], marker="^", markerfacecolor="white",
           markeredgecolor=COL["VisA"], lw=0, label="VisA"),
    Line2D([0], [0], marker="x", color=INK, lw=0, label="violated"),
    Line2D([0],[0], color="#4A4A4A", marker=r"$\rightarrow$", markersize=11,
           linestyle="--", label="empirical$\\to$raw transport"),
]
ax.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.995, 0.985),
          fontsize=8.3, frameon=False, ncol=1,
          handletextpad=0.5)
ax.set_title("Item count and observed FPR under alternative threshold rules",
             fontsize=10.5, pad=10)
fig.subplots_adjust(left=0.11, right=0.98, bottom=0.19, top=0.88)
save(fig, "fig1_capacity_operating")

# ---------------- Fig 2 ----------------
# 5x8 raw-arm state matrix (seeds 0-4 x 8 AD2 categories), from X2 (equivalence-gated).
raw_fp = {  # seed -> unit -> fp
 0: dict(zip(AD2_UNITS,[7,7,1,4,2,1,3,5])),
 1: dict(zip(AD2_UNITS,[6,5,1,5,0,1,3,6])),
 2: dict(zip(AD2_UNITS,[7,7,1,4,0,1,3,5])),
 3: dict(zip(AD2_UNITS,[7,8,1,4,2,1,3,5])),
 4: dict(zip(AD2_UNITS,[7,7,1,5,2,1,2,5])),
}
# empirical-arm violated cells (for marker): only wallplugs s2, s4
emp_viol = {(2,"wallplugs"),(4,"wallplugs")}
M = np.zeros((5,8))
ann = [["" for _ in range(8)] for _ in range(5)]
for i,s in enumerate(range(5)):
    for j,u in enumerate(AD2_UNITS):
        fp=raw_fp[s][u]; n0=ad2_n0[u]
        v = violated(fp,n0)
        M[i,j] = 1 if v else 0
        ann[i][j] = "%d/%d"%(fp,n0)
fig2, ax2 = plt.subplots(figsize=(7.4,3.5))
cmap = matplotlib.colors.ListedColormap(["#E2E2E2", "#666666"])
ax2.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
for i in range(5):
    for j in range(8):
        if M[i, j]:
            ax2.add_patch(Rectangle(
                (j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///",
                edgecolor="white", linewidth=0.0, zorder=4,
            ))
        ax2.text(j, i, ann[i][j], ha="center", va="center", fontsize=8.5,
                 color=("white" if M[i,j] else "k"), zorder=5)
        if (i, AD2_UNITS[j]) in emp_viol:  # empirical also violated here
            ax2.scatter(j+0.30, i-0.28, s=48, marker="o", facecolors="none",
                        edgecolors="black", linewidths=1.2, zorder=6)
ax2.set_xticks(range(8)); ax2.set_xticklabels(AD2_UNITS, rotation=30, ha="right", fontsize=8.5)
ax2.set_yticks(range(5)); ax2.set_yticklabels(["seed %d"%s for s in range(5)], fontsize=8.5)
ax2.set_xlabel("AD2 category (PatchCore, Test-M, $\\alpha=\\delta=5\\%$)", fontsize=9.5)
leg2 = [Patch(facecolor="#666666", edgecolor="white", hatch="///",
              label="raw mean+$3\\sigma$: violated"),
        Patch(facecolor="#E2E2E2", edgecolor="0.4", label="unresolved"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="black", markersize=7, linestyle="None",
               label="empirical also violated (corner circle)")]
ax2.legend(handles=leg2, loc="upper left", bbox_to_anchor=(1.01,1.0),
           fontsize=8.5, frameon=False)
ax2.set_title("PatchCore verdicts across seeds under raw mean+$3\\sigma$",
              fontsize=10.2)
fig2.tight_layout()
save(fig2, "fig2_multiseed_heatmap")


# ---------------- Cross-benchmark three-state breadth ----------------
state_rows = [
    ("AD2", 0, 0, 24),
    ("MVTec AD", 1, 13, 31),
    ("VisA", 1, 3, 32),
    ("BTAD", 0, 3, 6),
    ("MPDD", 0, 2, 16),
    ("MVTec LOCO", 1, 2, 12),
]
assert [sum(row[1:]) for row in state_rows] == [24, 45, 36, 9, 18, 15]
fig6, ax6 = plt.subplots(figsize=(7.0, 3.8))
y = np.arange(len(state_rows))
totals = np.array([sum(r[1:]) for r in state_rows], dtype=float)
certs = np.array([r[1] for r in state_rows], dtype=float)
viols = np.array([r[2] for r in state_rows], dtype=float)
unres = np.array([r[3] for r in state_rows], dtype=float)
segments = [(certs, GREEN, "certified", "///"),
            (viols, RED, "violated", "\\\\\\"),
            (unres, "#BFC6CC", "unresolved", "")]
left = np.zeros(len(state_rows))
for values, color, label, hatch in segments:
    widths = 100 * values / totals
    ax6.barh(y, widths, left=left, height=0.58, color=color,
             edgecolor="0.35", linewidth=0.6, hatch=hatch, label=label)
    for i, (w, raw) in enumerate(zip(widths, values.astype(int))):
        if raw and w >= 7:
            ax6.text(left[i] + w / 2, i, str(raw), ha="center", va="center",
                     fontsize=8.5, color=("white" if color in (GREEN, RED) else INK),
                     weight="bold")
    left += widths
for i, n in enumerate(totals.astype(int)):
    ax6.text(101.5, i, f"n={n}", ha="left", va="center", fontsize=8.5, color="#59656E")
ax6.set_yticks(y)
ax6.set_yticklabels([r[0] for r in state_rows])
ax6.invert_yaxis()
ax6.set_xlim(0, 109)
ax6.set_xlabel(r"share of marginal cells at $\alpha=\delta=5\%$  (%)")
ax6.set_title("Marginal three-state verdicts across datasets",
              pad=11)
ax6.legend(loc="lower center", bbox_to_anchor=(0.5, 1.19), ncol=3,
           frameon=False, fontsize=8.8)
ax6.grid(axis="x", color="#E8EBED", lw=0.7)
fig6.tight_layout(pad=1.2)
save(fig6, "fig6_breadth_states")


# ---------------- Lighting and domain shift mechanisms ----------------
fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(7.25, 3.55),
                                  gridspec_kw={"width_ratios": [0.9, 1.1]})
transitions = [(r"$U\to V$", 17, RED), (r"$U\to U$", 6, "#AEB7BE"),
               (r"$V\to V$", 1, "#7F2D25")]
bars = ax7a.barh(np.arange(3), [t[1] for t in transitions],
                 color=[t[2] for t in transitions], height=0.58,
                 hatch=["///", "", "\\\\\\"], edgecolor="0.35",
                 linewidth=0.6)
ax7a.set_yticks(np.arange(3))
ax7a.set_yticklabels([t[0] for t in transitions])
ax7a.invert_yaxis()
ax7a.set_xlim(0, 19)
ax7a.set_xlabel("matched cells (/24)")
ax7a.set_title(r"(a) AD2 lighting shift"+"\n"+r"Track C: Test-M $\to$ Test-S",
               fontsize=10.0)
for bar, (_, value, _) in zip(bars, transitions):
    ax7a.text(value + 0.3, bar.get_y() + bar.get_height()/2, str(value),
              va="center", fontsize=8.8, weight="bold", color=INK)
ax7a.text(0.98, 0.035, "U = unresolved; V = violated",
          transform=ax7a.transAxes, ha="right", va="bottom",
          fontsize=8.2, color="#65727C",
          bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.90))
ax7a.grid(axis="x", color="#E8EBED", lw=0.7)

models = ["EfficientAD", "PatchCore", "PaDiM"]
test_m = [0.0913, 0.2348, 0.1304]
test_s = [0.9867, 0.4400, 0.0267]
colors = [RED, BLUE, GREEN]
line_styles = ["-", "--", "-."]
markers = ["o", "s", "^"]
for model, m, s, color, linestyle, marker in zip(
        models, test_m, test_s, colors, line_styles, markers):
    ax7b.plot([0, 1], [m, s], marker=marker, lw=2.0, ms=5.5,
              color=color, ls=linestyle, markerfacecolor="white", label=model)
    start_dy = {"EfficientAD": -10, "PatchCore": 0, "PaDiM": 10}[model]
    end_dy = {"EfficientAD": 0, "PatchCore": 0, "PaDiM": -9}[model]
    ax7b.annotate(f"{m:.3f}", xy=(0, m), xytext=(-7, start_dy),
                  textcoords="offset points", ha="right", va="center",
                  fontsize=8.2, color=color)
    ax7b.annotate(f"{s:.3f}", xy=(1, s), xytext=(7, end_dy),
                  textcoords="offset points", ha="left", va="center",
                  fontsize=8.2, color=color)
ax7b.axhline(0.05, color=INK, ls=":", lw=1.0, alpha=0.9)
ax7b.text(0.55, 0.095, "5% budget", ha="center", va="bottom",
          fontsize=8.2, color=INK,
          bbox=dict(boxstyle="round,pad=0.13", fc="white", ec="none", alpha=0.90))
ax7b.set_xticks([0, 1])
ax7b.set_xticklabels(["Test-M", "Test-S"])
ax7b.set_xlim(-0.18, 1.18)
ax7b.set_ylim(0, 1.04)
ax7b.set_ylabel("descriptive item FPR")
ax7b.set_title("(b) AeBAD domain shift\nmodel-specific changes in operating point", fontsize=10.0)
ax7b.legend(loc="center left", bbox_to_anchor=(0.02, 0.72), fontsize=8.3,
            frameon=False)
ax7b.grid(axis="y", color="#E8EBED", lw=0.7)
fig7.suptitle("Operating-point changes under lighting and domain shift",
              y=1.02, fontsize=10.5, weight="bold", color=INK)
fig7.tight_layout()
save(fig7, "fig7_shift_mechanisms")


# ---------------- Reliability-demonstration extensions ----------------
artifact_dir = (
    Path(__file__).resolve().parent / "artifacts" / "reliability_extension"
)


def q_bounds(frames, alarms, items, maximum, threshold):
    if threshold == 1:
        if alarms == 0:
            return 0, 0
        return (
            max(1, int(np.ceil(alarms / maximum)), items - (frames - alarms)),
            min(alarms, items),
        )
    denominator = maximum - threshold + 1
    lower = max(0, int(np.ceil((alarms - (threshold - 1) * items) / denominator)))
    upper = min(
        items,
        alarms // threshold,
        (frames - items) // (threshold - 1),
    )
    return lower, upper


fig8, (ax8a, ax8b, ax8c) = plt.subplots(1, 3, figsize=(7.55, 3.12))

alarms = np.arange(56)
vote_styles = [(1, BLUE, "o"), (3, AMBER, "s"), (5, PURPLE, "^")]
for threshold, color, marker in vote_styles:
    bounds = np.asarray([q_bounds(55, int(a), 11, 5, threshold) for a in alarms])
    ax8a.plot(alarms, bounds[:, 0], color=color, lw=1.1, marker=marker,
              markevery=(4, 9), ms=3.0, markerfacecolor="white")
    ax8a.plot(alarms, bounds[:, 1], color=color, lw=1.1, ls="--",
              marker=marker, markevery=9, ms=3.0, markerfacecolor="white",
              label=rf"$q={threshold}$")
ax8a.set_xlim(0, 55)
ax8a.set_ylim(-0.3, 11.3)
ax8a.set_xlabel("alarming frames  $A$")
ax8a.set_ylabel("feasible rejected items  $K$")
ax8a.set_title("(a) Sharp vote geometry")
ax8a.legend(frameon=False, fontsize=8.2, ncol=1, loc="upper left")
ax8a.grid(color="#E8EBED", lw=0.6)

with (artifact_dir / "heterogeneity_sensitivity.csv").open(
    newline="", encoding="utf-8"
) as handle:
    heterogeneity = list(csv.DictReader(handle))
rho_styles = [(0.0, NAVY, "-", "o"), (0.01, TEAL, "--", "s"),
              (0.05, AMBER, "-.", "^"), (0.10, RED, ":", "D")]
for rho, color, linestyle, marker in rho_styles:
    selected = [row for row in heterogeneity
                if float(row["intraclass_correlation"]) == rho]
    selected.sort(key=lambda row: int(row["independent_batches"]))
    x = [int(row["independent_batches"]) for row in selected]
    y = [float(row["upper_coverage"]) for row in selected]
    ax8b.plot(x, y, marker=marker, ms=3.5, lw=1.25, color=color,
              ls=linestyle, markerfacecolor="white", label=rf"$\rho={rho:g}$")
ax8b.axhline(0.95, color=INK, ls=(0, (2, 1)), lw=1.0)
ax8b.set_xscale("log")
ax8b.set_xticks([1, 3, 6, 12, 60])
ax8b.set_xticklabels(["1", "3", "6", "12", "60"])
ax8b.set_ylim(0.64, 1.0)
ax8b.set_xlabel("independent batches")
ax8b.set_ylabel("CP upper-bound coverage")
ax8b.set_title("(b) Shared-batch sensitivity")
ax8b.legend(frameon=False, fontsize=8.2, loc="lower right")
ax8b.grid(axis="y", color="#E8EBED", lw=0.6)

with (artifact_dir / "dual_gate_summary.csv").open(
    newline="", encoding="utf-8"
) as handle:
    gate_rows = list(csv.DictReader(handle))
labels = ["EfficientAD", "PaDiM", "PatchCore"]
lookup = {row["model"]: row for row in gate_rows}
keys = ["efficientad", "padim", "patchcore"]
totals = np.asarray([int(lookup[key]["seeds"]) for key in keys], dtype=float)
parts = [
    ("release_ready", GREEN, "release-ready", "///"),
    ("continue_testing", "#BFC6CC", "continue", ""),
    ("not_release_ready", RED, "not ready", "\\\\\\"),
]
left = np.zeros(3)
for field, color, label, hatch in parts:
    raw = np.asarray([int(lookup[key][field]) for key in keys], dtype=float)
    widths = 100 * raw / totals
    ax8c.barh(np.arange(3), widths, left=left, height=0.55,
              color=color, edgecolor="0.35", lw=0.6, hatch=hatch,
              label=label)
    for idx, (start, width, value) in enumerate(zip(left, widths, raw.astype(int))):
        if value and width >= 12:
            ax8c.text(start + width / 2, idx, str(value), ha="center", va="center",
                      fontsize=8.2, color=("white" if color in (GREEN, RED) else INK),
                      weight="bold")
    left += widths
ax8c.set_yticks(np.arange(3))
ax8c.set_yticklabels(labels)
ax8c.invert_yaxis()
ax8c.set_xlim(0, 100)
ax8c.set_xlabel("pipeline seeds  (%)")
ax8c.set_title("(c) Dual-risk release gate")
ax8c.legend(frameon=False, fontsize=8.2, loc="lower center",
            bbox_to_anchor=(0.5, 1.18), ncol=3, columnspacing=1.0)
ax8c.grid(axis="x", color="#E8EBED", lw=0.6)

fig8.tight_layout(w_pad=1.1)
save(fig8, "fig8_reliability_extensions")
