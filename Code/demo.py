"""
VGG Architecture: Teaching & Visualization Tool
================================================
A complete Streamlit application that teaches and visualizes VGG CNNs.
- Section A: Live Feature Map Explorer (VGG-16 & VGG-19 with real images)
- Section B: The VGG Story – The Power of 3×3 (interactive narrative)

Run with: streamlit run app.py
"""

# ── Matplotlib MUST be set before any other matplotlib/pyplot import ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import streamlit as st
import numpy as np
import io
import time
import copy
import json
import urllib.request
from PIL import Image

import torch
import torchvision.models as models
import torchvision.transforms as transforms

# ── Force CPU globally so no CUDA calls are ever made ─────────────────
DEVICE = torch.device("cpu")

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="VGG Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────
GLOBAL_CSS = """
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: #f8fafc !important;
    color: #1e293b !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3a5f 0%, #0f172a 100%) !important;
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { color: #cbd5e1 !important; font-size: 0.95rem; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #7dd3fc !important; }

.hero-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #1d4ed8 50%, #7c3aed 100%);
    border-radius: 16px; padding: 2.5rem 2rem; margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(30,58,95,0.18);
}
.hero-header h1 { color: #fff !important; font-size: 2.4rem; font-weight: 800; margin: 0; }
.hero-header p  { color: #bae6fd !important; font-size: 1.05rem; margin: 0.5rem 0 0; }

.card {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px;
    padding: 1.4rem 1.6rem; margin-bottom: 1rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.card-blue   { border-left: 5px solid #3b82f6; }
.card-purple { border-left: 5px solid #8b5cf6; }
.card-green  { border-left: 5px solid #10b981; }
.card-amber  { border-left: 5px solid #f59e0b; }
.card-red    { border-left: 5px solid #ef4444; }

.section-title {
    font-size: 1.55rem; font-weight: 700; color: #1e3a5f;
    border-bottom: 3px solid #3b82f6;
    padding-bottom: 0.35rem; margin-bottom: 1rem;
}
.subsection-title {
    font-size: 1.15rem; font-weight: 600; color: #334155; margin: 1rem 0 0.4rem;
}

.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: 0.8rem 0; }
.metric-box {
    flex: 1; min-width: 140px;
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    border: 1px solid #bfdbfe; border-radius: 12px;
    padding: 0.9rem 1rem; text-align: center;
}
.metric-box .val { font-size: 1.6rem; font-weight: 800; color: #1d4ed8; }
.metric-box .lbl { font-size: 0.78rem; color: #64748b; margin-top: 2px; }
.metric-box-green  { background: linear-gradient(135deg,#f0fdf4,#dcfce7) !important;
                     border-color:#86efac !important; }
.metric-box-green .val  { color: #16a34a !important; }
.metric-box-purple { background: linear-gradient(135deg,#faf5ff,#ede9fe) !important;
                     border-color:#c4b5fd !important; }
.metric-box-purple .val { color: #7c3aed !important; }
.metric-box-amber  { background: linear-gradient(135deg,#fffbeb,#fef3c7) !important;
                     border-color:#fcd34d !important; }
.metric-box-amber .val  { color: #d97706 !important; }

.step-indicator { display: flex; gap: 6px; align-items: center; margin-bottom: 1.2rem; }
.step-dot {
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 700;
    background: #e2e8f0; color: #64748b; transition: all 0.2s;
}
.step-dot.active { background: #1d4ed8; color: #fff; }
.step-dot.done   { background: #10b981; color: #fff; }
.step-line { flex: 1; height: 3px; background: #e2e8f0; border-radius: 2px; }
.step-line.done { background: #10b981; }

.callout {
    background: linear-gradient(135deg, #eff6ff, #f0fdf4);
    border: 1px solid #bfdbfe; border-radius: 10px;
    padding: 0.9rem 1.1rem; margin: 0.8rem 0;
    font-size: 0.92rem; color: #1e3a5f; line-height: 1.6;
}
.callout strong { color: #1d4ed8; }
.callout-purple { background: linear-gradient(135deg,#faf5ff,#ede9fe) !important;
    border-color:#c4b5fd !important; color:#4c1d95 !important; }
.callout-purple strong { color: #7c3aed !important; }
.callout-green  { background: linear-gradient(135deg,#f0fdf4,#dcfce7) !important;
    border-color:#86efac !important; color:#14532d !important; }
.callout-green strong  { color: #16a34a !important; }

.footer {
    margin-top: 3rem; padding: 1.5rem; text-align: center;
    background: #1e3a5f; border-radius: 14px; color: #94a3b8; font-size: 0.85rem;
}
.footer strong { color: #7dd3fc; }

table.cmp-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
table.cmp-table th {
    background: #1e3a5f; color: #fff;
    padding: 0.55rem 0.75rem; text-align: left;
}
table.cmp-table td {
    padding: 0.45rem 0.75rem;
    border-bottom: 1px solid #e2e8f0; color: #1e293b;
}
table.cmp-table tr:nth-child(even) td { background: #f8fafc; }

.arch-flow {
    display: flex; align-items: flex-start; flex-wrap: wrap;
    gap: 6px; padding: 1rem 0; overflow: visible;
}
.arch-box {
    border-radius: 12px; padding: 0.75rem 0.6rem; width: 105px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.08);
    text-align: center; border: 2px solid; flex-shrink: 0;
}
.arch-box .arch-label { font-weight: 800; font-size: 0.82rem; margin-bottom: 6px; }
.arch-box .arch-op {
    font-size: 0.64rem; padding: 2px 3px;
    background: rgba(255,255,255,0.7); border-radius: 4px;
    margin: 2px 0; display: block; word-break: break-word;
}
.arch-box .arch-sub {
    font-size: 0.6rem; color: #64748b; font-weight: 600; margin-top: 6px;
}
.arch-arrow {
    font-size: 1.3rem; color: #94a3b8; padding-top: 2rem; flex-shrink: 0;
}

.bar-wrap { margin: 8px 0; }
.bar-row-label { display: flex; justify-content: space-between; margin-bottom: 3px; }
.bar-row-label span { font-size: 0.85rem; font-weight: 600; }
.bar-bg  { background: #e2e8f0; border-radius: 6px; height: 14px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 6px; }

.dot-product-box {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 0.6rem 0.8rem;
    font-family: 'IBM Plex Mono', 'Courier New', monospace;
    font-size: 0.82rem; line-height: 2.1;
    max-height: 240px; overflow-y: auto; overflow-x: hidden;
}

div[data-testid="stFileUploader"] {
    border: 2px dashed #3b82f6 !important;
    border-radius: 12px !important; padding: 1rem !important;
}
.stButton button { border-radius: 8px !important; font-weight: 600 !important; }
section[data-testid="stMain"] > div { overflow-x: visible !important; }
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# IMAGENET LABELS
# ──────────────────────────────────────────────
IMAGENET_LABELS_URL = (
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/"
    "master/imagenet-simple-labels.json"
)

@st.cache_data(show_spinner=False)
def load_imagenet_labels() -> list:
    try:
        with urllib.request.urlopen(IMAGENET_LABELS_URL, timeout=10) as r:
            return json.loads(r.read().decode())
    except Exception:
        return [f"class_{i}" for i in range(1000)]

# ──────────────────────────────────────────────
# LAZY MODEL LOADING — one model at a time
# ──────────────────────────────────────────────
# Each model is cached independently under its own key.
# VGG-16 and VGG-19 are NEVER loaded at the same time unless
# the user explicitly switches between them in the sidebar.
# All tensors and parameters are kept on CPU.

@st.cache_resource(show_spinner="Loading VGG-16 weights…")
def load_vgg16_cpu():
    """Download (once) and cache VGG-16 fully on CPU."""
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    model = model.to(DEVICE)
    model.eval()
    return model

@st.cache_resource(show_spinner="Loading VGG-19 weights…")
def load_vgg19_cpu():
    """Download (once) and cache VGG-19 fully on CPU."""
    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    model = model.to(DEVICE)
    model.eval()
    return model

def get_model(name: str):
    """Return the cached model for *name* ('VGG-16' or 'VGG-19').
    Only the requested model is ever loaded into memory."""
    if name == "VGG-16":
        return load_vgg16_cpu()
    return load_vgg19_cpu()

# ──────────────────────────────────────────────
# LAYER MAPS
# ──────────────────────────────────────────────
VGG16_LAYER_MAP = {
    "Block 1 – Conv1 (64ch)":   0, "Block 1 – Conv2 (64ch)":   2,
    "Block 1 – MaxPool":        4, "Block 2 – Conv1 (128ch)":  5,
    "Block 2 – Conv2 (128ch)":  7, "Block 2 – MaxPool":        9,
    "Block 3 – Conv1 (256ch)": 10, "Block 3 – Conv2 (256ch)": 12,
    "Block 3 – Conv3 (256ch)": 14, "Block 3 – MaxPool":       16,
    "Block 4 – Conv1 (512ch)": 17, "Block 4 – Conv2 (512ch)": 19,
    "Block 4 – Conv3 (512ch)": 21, "Block 4 – MaxPool":       23,
    "Block 5 – Conv1 (512ch)": 24, "Block 5 – Conv2 (512ch)": 26,
    "Block 5 – Conv3 (512ch)": 28, "Block 5 – MaxPool":       30,
}
VGG19_LAYER_MAP = {
    "Block 1 – Conv1 (64ch)":   0, "Block 1 – Conv2 (64ch)":   2,
    "Block 1 – MaxPool":        4, "Block 2 – Conv1 (128ch)":  5,
    "Block 2 – Conv2 (128ch)":  7, "Block 2 – MaxPool":        9,
    "Block 3 – Conv1 (256ch)": 10, "Block 3 – Conv2 (256ch)": 12,
    "Block 3 – Conv3 (256ch)": 14, "Block 3 – Conv4 (256ch)": 16,
    "Block 3 – MaxPool":       18, "Block 4 – Conv1 (512ch)": 19,
    "Block 4 – Conv2 (512ch)": 21, "Block 4 – Conv3 (512ch)": 23,
    "Block 4 – Conv4 (512ch)": 25, "Block 4 – MaxPool":       27,
    "Block 5 – Conv1 (512ch)": 28, "Block 5 – Conv2 (512ch)": 30,
    "Block 5 – Conv3 (512ch)": 32, "Block 5 – Conv4 (512ch)": 34,
    "Block 5 – MaxPool":       36,
}

def get_layer_map(name: str) -> dict:
    return VGG16_LAYER_MAP if name == "VGG-16" else VGG19_LAYER_MAP

# ──────────────────────────────────────────────
# IMAGE PRE-PROCESSING  (cached per image bytes)
# ──────────────────────────────────────────────
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

@st.cache_data(show_spinner=False)
def preprocess_image_bytes(raw_bytes: bytes) -> torch.Tensor:
    """Convert raw uploaded bytes → normalised CPU tensor (1,3,224,224)."""
    from io import BytesIO
    img = Image.open(BytesIO(raw_bytes)).convert("RGB")
    return _TRANSFORM(img).unsqueeze(0).to(DEVICE)

# ──────────────────────────────────────────────
# INFERENCE HELPERS  (CPU-explicit)
# ──────────────────────────────────────────────
def extract_activations(model, tensor: torch.Tensor, layer_idx: int):
    """Hook into model.features[layer_idx] and return activation numpy array."""
    activation: dict = {}

    def _hook(module, inp, out):
        activation["feat"] = out.detach().cpu()

    handle = model.features[layer_idx].register_forward_hook(_hook)
    with torch.no_grad():
        _ = model(tensor.to(DEVICE))
    handle.remove()

    feat = activation.get("feat")
    return feat[0].numpy() if feat is not None else None

def run_full_inference(model, tensor: torch.Tensor) -> np.ndarray:
    """Return softmax probabilities as a numpy array (CPU)."""
    with torch.no_grad():
        logits = model(tensor.to(DEVICE))
    return torch.softmax(logits, dim=1)[0].cpu().numpy()

# ──────────────────────────────────────────────
# MATPLOTLIB HELPERS
# ──────────────────────────────────────────────
def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()

def render_feature_maps(activations, n_maps: int, title: str,
                         colormap: str = "viridis"):
    n_maps  = min(n_maps, activations.shape[0])
    ncols   = min(8, n_maps)
    nrows   = (n_maps + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols*1.9, nrows*1.9),
                              facecolor="#f8fafc")
    fig.suptitle(title, fontsize=10, fontweight="bold", color="#1e3a5f", y=1.01)
    flat = ([axes] if n_maps == 1
            else list(axes) if nrows == 1
            else list(np.array(axes).reshape(-1)))
    for i, ax in enumerate(flat):
        if i < n_maps:
            ax.imshow(activations[i], cmap=colormap, aspect="auto")
            ax.set_title(f"Ch {i}", fontsize=7, color="#334155")
        ax.axis("off")
    plt.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════════
# SECTION B — HARMONIOUS MUTED PALETTE
# ══════════════════════════════════════════════════════════════════
_P = {
    "input":    dict(cmap="Blues",   cmap_lo=0.12, cmap_hi=0.68,
                     border="#a8bdd4", hi_edge="#c0392b", figbg="#eef2f7"),
    "pad":      dict(cmap="Greys",   cmap_lo=0.08, cmap_hi=0.55,
                     border="#b0b8c4", hi_edge="#c0392b", figbg="#f0f2f5"),
    "filter":   dict(cmap="RdPu",    cmap_lo=0.08, cmap_hi=0.55,
                     border="#c4a4b8", hi_edge="#b45309", figbg="#f7f0f5"),
    "conv_out": dict(cmap="GnBu",    cmap_lo=0.10, cmap_hi=0.65,
                     border="#8db8c8", hi_edge="#c0392b", figbg="#edf4f7"),
    "relu_out": dict(cmap="YlGn",    cmap_lo=0.12, cmap_hi=0.62,
                     border="#97bda0", hi_edge="#c0392b", figbg="#eef5f0"),
    "pool_out": dict(cmap="YlOrBr",  cmap_lo=0.10, cmap_hi=0.58,
                     border="#c8b080", hi_edge="#c0392b", figbg="#f7f3eb"),
    "highlight":dict(cmap="OrRd",    cmap_lo=0.10, cmap_hi=0.58,
                     border="#c89080", hi_edge="#7c3aed", figbg="#f7efeb"),
    "rf":       dict(cmap="PuBuGn",  cmap_lo=0.10, cmap_hi=0.65,
                     border="#8ab8b0", hi_edge="#c0392b", figbg="#edf5f3"),
}

# Five harmonious desaturated accent colours used in HTML elements
_C = {
    "slate": "#4a6785", "teal":  "#4a8080", "sage":  "#5a7a5a",
    "sand":  "#8a7050", "rose":  "#8a5060", "text":  "#2c3e50",
    "sub":   "#64748b", "arrow": "#94a3b8", "bg_box":"#f0f4f8",
}


def render_matrix_png(
    mat: np.ndarray,
    palette: str = "input",
    highlight_cells: set = None,
    title: str = "",
    fig_scale: float = 0.65,
) -> bytes:
    """Render a 2-D array as a crisp PNG using the muted palette."""
    if highlight_cells is None:
        highlight_cells = set()

    p      = _P.get(palette, _P["input"])
    nrows, ncols = mat.shape
    fig_w  = ncols * fig_scale + 0.35
    fig_h  = nrows * fig_scale + (0.50 if title else 0.14)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=p["figbg"])
    ax.set_facecolor(p["figbg"])

    vmin, vmax = float(mat.min()), float(mat.max())
    vrange = max(vmax - vmin, 1e-6)
    cmap   = plt.get_cmap(p["cmap"])
    gap, r_box = 0.07, 0.11

    for r in range(nrows):
        for c in range(ncols):
            v = mat[r, c]
            t = p["cmap_lo"] + (v - vmin) / vrange * (p["cmap_hi"] - p["cmap_lo"])
            colour  = cmap(t)
            is_hi   = (r, c) in highlight_cells
            edge_c  = p["hi_edge"] if is_hi else p["border"]
            lw      = 2.2 if is_hi else 0.9

            x0 = c + gap / 2
            y0 = (nrows - 1 - r) + gap / 2
            w  = h = 1 - gap

            ax.add_patch(FancyBboxPatch(
                (x0, y0), w, h,
                boxstyle=f"round,pad=0,rounding_size={r_box}",
                facecolor=colour, edgecolor=edge_c, linewidth=lw, zorder=1,
            ))

            label = (f"{v:.2f}" if abs(v) < 10
                     else f"{v:.1f}" if abs(v) < 100
                     else str(int(round(v))))
            lum   = 0.299*colour[0] + 0.587*colour[1] + 0.114*colour[2]
            ax.text(x0 + w/2, y0 + h/2, label,
                    ha="center", va="center",
                    fontsize=max(7, min(13, int(fig_scale * 18))),
                    fontweight="bold", fontfamily="monospace",
                    color="#2c3e50" if lum > 0.50 else "#f0f4f8",
                    zorder=2)

    ax.set_xlim(0, ncols); ax.set_ylim(0, nrows)
    ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", color="#34495e", pad=6)
    plt.tight_layout(pad=0.2)
    return fig_to_bytes(fig)


# ══════════════════════════════════════════════════════════════════
# SECTION A — Live Feature Map Explorer
# ══════════════════════════════════════════════════════════════════
def section_feature_explorer(selected_model: str):
    """
    Section A.
    `selected_model` is passed in from the sidebar so only ONE model
    is ever resident in memory at a time.
    """
    st.markdown('<div class="section-title">🔬 Live Feature Map Explorer</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card card-blue">
    <b>Model: {selected_model}</b> — Upload an image to explore how this model
    transforms it through its convolutional layers.
    Change the model in the sidebar at any time.
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Choose an image (JPG or PNG)",
                                 type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("👆 Upload an image to begin exploring VGG activations.")
        _show_hook_expander()
        return

    # ── Cache pre-processing keyed on raw bytes ────────────────────
    raw_bytes = uploaded.read()
    try:
        tensor = preprocess_image_bytes(raw_bytes)
    except Exception as e:
        st.error(f"Could not process image: {e}")
        return

    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    layer_map = get_layer_map(selected_model)

    col_img, col_ctrl = st.columns([1, 2], gap="large")
    with col_img:
        st.image(pil_img, caption="Uploaded Image", use_container_width=True)
    with col_ctrl:
        st.markdown('<div class="subsection-title">⚙️ Explorer Controls</div>',
                    unsafe_allow_html=True)
        # ── Mode — no simultaneous dual-model loading ──────────────
        mode = st.radio("Mode", [
            "Single Model",
            "Before / After MaxPool",
        ], index=0)

        layer_name = st.selectbox("Select Layer", list(layer_map), index=2)
        n_maps     = st.slider("Feature Maps to Show", 4, 64, 16, 4)
        cmap_choice = st.selectbox("Colormap",
                                    ["viridis", "plasma", "inferno",
                                     "magma", "coolwarm", "turbo"])

    # ── Lazy-load only the selected model ─────────────────────────
    model = get_model(selected_model)

    if mode == "Single Model":
        with st.spinner(f"Running {selected_model} forward pass…"):
            acts  = extract_activations(model, tensor, layer_map[layer_name])
            probs = run_full_inference(model, tensor)
        if acts is None:
            st.error("Hook failed — try a different layer."); return

        _show_activation_stats(acts)
        with st.spinner("Rendering feature maps…"):
            fig = render_feature_maps(acts, n_maps,
                                      f"{selected_model} — {layer_name}",
                                      cmap_choice)
            st.image(fig_to_bytes(fig), use_container_width=True)
            plt.close(fig)
        _show_top3_predictions(probs)

    else:  # Before / After MaxPool
        pool_layers = {k: v for k, v in layer_map.items() if "MaxPool" in k}
        pool_name   = st.selectbox("Select MaxPool Layer", list(pool_layers))
        channel_idx = st.slider("Channel to Inspect", 0, 31, 0)
        pool_idx    = layer_map[pool_name]

        with st.spinner("Extracting before/after pooling activations…"):
            acts_pre  = extract_activations(model, tensor, pool_idx - 1)
            acts_post = extract_activations(model, tensor, pool_idx)

        if acts_pre is None or acts_post is None:
            st.error("Could not extract activations."); return

        ch = min(channel_idx, acts_pre.shape[0]-1, acts_post.shape[0]-1)
        st.markdown(f'<div class="subsection-title">Before vs After: {pool_name}</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Before MaxPool** — channel {ch}")
            st.caption(f"Shape: {acts_pre.shape[1]}×{acts_pre.shape[2]}")
            fig = _single_channel_heatmap(acts_pre[ch], "Before (ReLU out)", "plasma")
            st.image(fig_to_bytes(fig), use_container_width=True); plt.close(fig)
        with c2:
            st.markdown(f"**After MaxPool** — channel {ch}")
            st.caption(f"Shape: {acts_post.shape[1]}×{acts_post.shape[2]}")
            fig = _single_channel_heatmap(acts_post[ch], "After MaxPool", "viridis")
            st.image(fig_to_bytes(fig), use_container_width=True); plt.close(fig)
        st.markdown("""
        <div class="callout"><strong>Key Insight:</strong> After MaxPool the spatial
        resolution halves but the strongest activations are preserved.</div>""",
        unsafe_allow_html=True)
        _show_top3_predictions(run_full_inference(model, tensor))

    _show_hook_expander()


# ── Section A helpers ─────────────────────────────────────────────
def _single_channel_heatmap(data, title, cmap):
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="#f8fafc")
    im = ax.imshow(data, cmap=cmap, aspect="auto")
    ax.set_title(title, fontsize=9, color="#1e3a5f", fontweight="bold")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig

def _show_activation_stats(acts):
    mean_v   = float(np.mean(acts))
    std_v    = float(np.std(acts))
    max_v    = float(np.max(acts))
    sparsity = float(np.mean(acts == 0) * 100)
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-box">
        <div class="val">{mean_v:.4f}</div><div class="lbl">Mean Activation</div>
      </div>
      <div class="metric-box metric-box-purple">
        <div class="val">{std_v:.4f}</div><div class="lbl">Std Deviation</div>
      </div>
      <div class="metric-box metric-box-green">
        <div class="val">{max_v:.3f}</div><div class="lbl">Max Activation</div>
      </div>
      <div class="metric-box metric-box-amber">
        <div class="val">{sparsity:.1f}%</div><div class="lbl">Sparsity (zeros)</div>
      </div>
    </div>""", unsafe_allow_html=True)

def _show_top3_predictions(probs):
    labels   = load_imagenet_labels()
    top3_idx = np.argsort(probs)[::-1][:3]
    st.markdown('<div class="subsection-title">🏆 Top-3 ImageNet Predictions</div>',
                unsafe_allow_html=True)
    colors = ["#1d4ed8", "#7c3aed", "#0891b2"]
    html   = ""
    for rank, (idx, color) in enumerate(zip(top3_idx, colors)):
        lbl = labels[idx] if idx < len(labels) else f"class_{idx}"
        pct = probs[idx] * 100
        html += f"""
        <div style="margin:8px 0">
          <div style="display:flex;justify-content:space-between;margin-bottom:3px">
            <span style="font-weight:600;color:#1e293b">
              #{rank+1} {lbl.replace('_',' ').title()}</span>
            <span style="font-weight:800;color:{color}">{pct:.2f}%</span>
          </div>
          <div style="background:#e2e8f0;border-radius:6px;height:14px;overflow:hidden">
            <div style="width:{min(pct,100):.1f}%;background:{color};
                        height:100%;border-radius:6px"></div>
          </div>
        </div>"""
    st.markdown(f'<div class="card card-green">{html}</div>', unsafe_allow_html=True)

def _show_hook_expander():
    with st.expander("🔧 How Forward Hooks Work (Annotated Code)"):
        st.markdown("""
        <div class="card card-purple">
        <b>The Hook Mechanism</b> — PyTorch lets you intercept any layer's output
        during a forward pass with <code>register_forward_hook</code>.
        </div>""", unsafe_allow_html=True)
        st.code('''
# All tensors stay on CPU — no .cuda() calls anywhere.
activation = {}
def hook_fn(module, input, output):
    activation["feat"] = output.detach().cpu()

handle = model.features[layer_idx].register_forward_hook(hook_fn)
with torch.no_grad():
    _ = model(tensor)   # tensor already on CPU
handle.remove()

feat_maps = activation["feat"][0].numpy()  # shape: [C, H, W]
        ''', language="python")
        st.markdown("""
        **Memory note:** Only the model selected in the sidebar is ever loaded.
        `@st.cache_resource` keeps it alive between reruns without reloading.
        """)


# ══════════════════════════════════════════════════════════════════
# SECTION B DATA
# ══════════════════════════════════════════════════════════════════
STAR_8x8 = np.array([
    [0,  0,  0,  4,  4,  0,  0,  0],
    [0,  0,  4,  8,  8,  4,  0,  0],
    [0,  4,  8, 16, 16,  8,  4,  0],
    [4,  8, 16, 32, 32, 16,  8,  4],
    [4,  8, 16, 32, 32, 16,  8,  4],
    [0,  4,  8, 16, 16,  8,  4,  0],
    [0,  0,  4,  8,  8,  4,  0,  0],
    [0,  0,  0,  4,  4,  0,  0,  0],
], dtype=float)

EDGE_FILTER = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=float)
BLUR_FILTER = np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=float) / 16.0

def _apply_conv(img, filt, pad=1):
    padded = np.pad(img, pad, mode="constant", constant_values=0)
    H, W   = img.shape
    out    = np.zeros((H, W))
    fh, fw = filt.shape
    for r in range(H):
        for c in range(W):
            out[r, c] = np.sum(padded[r:r+fh, c:c+fw] * filt)
    return out

def _relu(x):           return np.maximum(0, x)

def _maxpool2x2(x):
    H, W = x.shape
    out  = np.zeros((H//2, W//2))
    for r in range(H//2):
        for c in range(W//2):
            out[r, c] = x[2*r:2*r+2, 2*c:2*c+2].max()
    return out

def render_step_indicator(current: int, total: int = 6) -> str:
    html = '<div class="step-indicator">'
    for i in range(total):
        if i > 0:
            cls = "done" if i <= current else ""
            html += f'<div class="step-line {cls}"></div>'
        cls  = "done" if i < current else ("active" if i == current else "")
        html += f'<div class="step-dot {cls}">{i}</div>'
    html += "</div>"
    return html


# ══════════════════════════════════════════════════════════════════
# STEP 0 – The 8×8 Image
# ══════════════════════════════════════════════════════════════════
def story_step0():
    st.markdown("""
    <div class="card card-blue">
    <b>Step 0 of 5:</b> Meet our tiny 8×8 "star" image — a simplified stand-in for any
    real image. The same principles apply to 224×224 ImageNet photos.
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown('<div class="subsection-title">Our 8×8 Input Image</div>',
                    unsafe_allow_html=True)
        st.image(render_matrix_png(STAR_8x8, palette="input", fig_scale=0.72),
                 use_container_width=True)
        st.caption("Values: pixel intensities (0 = black, 32 = brightest).")

    with col2:
        st.markdown('<div class="subsection-title">What is this?</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="callout">
        <strong>Think of each number as a pixel brightness.</strong><br><br>
        Real images have three channels (RGB) and are 224×224 pixels for VGG.
        We use 8×8 and one channel to make every step visible and understandable.<br><br>
        The star pattern has a bright centre (32) fading to 0 at the edges.
        A CNN's job: detect such patterns at different scales and positions.
        </div>""", unsafe_allow_html=True)

        # Tinted visual preview matching the Blues palette
        fig, ax = plt.subplots(figsize=(3.2, 3.2), facecolor="#eef2f7")
        ax.set_facecolor("#eef2f7")
        cmap_prev = plt.get_cmap("Blues")
        norm_star = STAR_8x8 / 32.0
        for r in range(8):
            for c in range(8):
                t   = 0.12 + norm_star[r, c] * 0.56
                col = cmap_prev(t)
                ax.add_patch(FancyBboxPatch(
                    (c+0.05, 7-r+0.05), 0.90, 0.90,
                    boxstyle="round,pad=0,rounding_size=0.11",
                    facecolor=col, edgecolor="#a8bdd4",
                    linewidth=0.8, zorder=1,
                ))
        ax.set_xlim(0, 8); ax.set_ylim(0, 8)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title("Visual Preview", fontsize=10,
                     color="#34495e", fontweight="bold", pad=7)
        plt.tight_layout(pad=0.3)
        st.image(fig_to_bytes(fig), use_container_width=True)
        plt.close(fig)

    st.markdown("""
    <div class="callout callout-purple">
    <strong>Goal:</strong> Over the next 5 steps we'll apply convolutions, study receptive
    fields, measure parameter savings, and build up to the full VGG-16 architecture.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# STEP 1 – Single 3×3 Convolution
# ══════════════════════════════════════════════════════════════════
def story_step1():
    st.markdown("""
    <div class="card card-purple">
    <b>Step 1 of 5:</b> Apply one 3×3 convolution to the 8×8 image.
    Watch the filter slide across the padded image and compute the output.
    </div>""", unsafe_allow_html=True)

    padded   = np.pad(STAR_8x8, 1, mode="constant", constant_values=0)
    conv_out = _apply_conv(STAR_8x8, EDGE_FILTER)
    relu_out = _relu(conv_out)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown('<div class="subsection-title">3×3 Edge Filter</div>',
                    unsafe_allow_html=True)
        st.image(render_matrix_png(EDGE_FILTER, "filter", fig_scale=1.1),
                 use_container_width=False)
        st.caption("Highlights edges by subtracting surrounding pixels.")
    with col2:
        st.markdown('<div class="subsection-title">Zero-Padded Input (10×10)</div>',
                    unsafe_allow_html=True)
        st.image(render_matrix_png(padded, "pad", fig_scale=0.58),
                 use_container_width=True)
        st.caption("Padding=1 keeps output 8×8. VGG always uses padding=1.")

    st.markdown('<div class="subsection-title">▶ Convolution Animation</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="callout" style="margin-bottom:0.8rem">
    The filter slides <strong>left-to-right, top-to-bottom</strong>. At every position
    it extracts a 3×3 patch, performs an <b>element-wise multiply</b>, then <b>sums</b>
    all 9 products → one output value.
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        play = st.button("▶ Play Convolution", key="play_conv1",
                         use_container_width=True)
    with c2:
        stop = st.button("⏹ Stop", key="stop_conv1", use_container_width=True)
    with c3:
        speed_ms = st.slider("Speed (ms/step)", 30, 500, 150, 10, key="conv1_speed")
        label = ("Very Fast" if speed_ms <= 60 else "Fast" if speed_ms <= 150
                 else "Medium" if speed_ms <= 300 else "Slow")
        st.caption(f"Speed: **{label}**")

    if stop: st.session_state["conv1_stop"] = True
    if play: st.session_state["conv1_stop"] = False

    anim_ph = st.empty()

    if play and not st.session_state.get("conv1_stop", False):
        partial_out = np.zeros((8, 8))
        for fr in range(8):
            for fc in range(8):
                if st.session_state.get("conv1_stop", False): break
                patch   = padded[fr:fr+3, fc:fc+3]
                dot_sum = float(np.sum(patch * EDGE_FILTER))
                partial_out[fr, fc] = conv_out[fr, fc]

                dp_lines = []
                for dr in range(3):
                    for dc in range(3):
                        prod    = patch[dr, dc] * EDGE_FILTER[dr, dc]
                        col_str = "#3d6b52" if prod >= 0 else "#8a3a4a"
                        dp_lines.append(
                            f'<span style="color:{col_str};font-weight:700">'
                            f'{patch[dr,dc]:.0f}×({EDGE_FILTER[dr,dc]:.0f})'
                            f'={prod:.1f}</span>'
                        )
                dp_html  = "<br>".join(dp_lines)
                sum_col  = "#4a6785" if dot_sum >= 0 else "#8a5060"
                dp_html += (
                    f'<hr style="border:1px solid #e2e8f0;margin:4px 0">'
                    f'<span style="font-weight:800;color:{sum_col};font-size:0.9rem">'
                    f'Sum = {dot_sum:.1f}</span>'
                )

                hi_pad = {(fr+dr, fc+dc) for dr in range(3) for dc in range(3)}
                hi_out = {(fr, fc)}

                with anim_ph.container():
                    a1, a2, a3 = st.columns([1.1, 0.85, 1.4])
                    with a1:
                        st.caption(f"Padded Input — window [{fr},{fc}]")
                        st.image(render_matrix_png(padded, "pad",
                                                   highlight_cells=hi_pad,
                                                   fig_scale=0.48),
                                 use_container_width=True)
                    with a2:
                        st.caption("Filter")
                        st.image(render_matrix_png(EDGE_FILTER, "filter",
                                                   fig_scale=0.85),
                                 use_container_width=False)
                        st.markdown(
                            f"<div style='text-align:center;font-size:1.4rem;"
                            f"color:{_C['sub']}'>⊙</div>",
                            unsafe_allow_html=True)
                        st.caption(f"Patch [{fr},{fc}]")
                        st.image(render_matrix_png(patch, "pad", fig_scale=0.85),
                                 use_container_width=False)
                    with a3:
                        st.caption("Dot Product")
                        st.markdown(
                            f'<div class="dot-product-box">{dp_html}</div>',
                            unsafe_allow_html=True)
                        st.caption(f"Output — [{fr},{fc}] filled")
                        st.image(render_matrix_png(partial_out, "conv_out",
                                                   highlight_cells=hi_out,
                                                   fig_scale=0.48),
                                 use_container_width=True)
                time.sleep(speed_ms / 1000.0)
        anim_ph.empty()

    st.markdown('<div class="subsection-title">Convolution Output & ReLU</div>',
                unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    with r1:
        st.caption("Input (8×8)")
        st.image(render_matrix_png(STAR_8x8, "input", fig_scale=0.62),
                 use_container_width=True)
    with r2:
        st.caption("Conv Output (8×8)")
        st.image(render_matrix_png(conv_out, "conv_out", fig_scale=0.62),
                 use_container_width=True)
    with r3:
        st.caption("After ReLU (8×8)")
        st.image(render_matrix_png(relu_out, "relu_out", fig_scale=0.62),
                 use_container_width=True)

    st.markdown("""
    <div class="callout callout-green">
    <strong>Key Takeaway:</strong> With padding=1 the output stays 8×8. VGG uses this
    everywhere — spatial size only halves at MaxPool. ReLU zeroes negatives.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# STEP 2 – Two 3×3 = 5×5 RF
# ══════════════════════════════════════════════════════════════════
def story_step2():
    st.markdown("""
    <div class="card card-green">
    <b>Step 2 of 5:</b> Stack <em>two</em> 3×3 convolutions.
    The output pixel "sees" a 5×5 region.
    </div>""", unsafe_allow_html=True)

    conv1 = _relu(_apply_conv(STAR_8x8, EDGE_FILTER))
    conv2 = _relu(_apply_conv(conv1,    EDGE_FILTER))

    st.markdown('<div class="subsection-title">Two-Pass Feature Maps</div>',
                unsafe_allow_html=True)
    mc1, arr1, mc2, arr2, mc3 = st.columns([3, 0.4, 3, 0.4, 3])
    with mc1:
        st.caption("Input (8×8)")
        st.image(render_matrix_png(STAR_8x8, "input", fig_scale=0.62),
                 use_container_width=True)
    with arr1:
        st.markdown(f"<div style='font-size:2rem;color:{_C['arrow']};"
                    f"padding-top:2.5rem;text-align:center'>→</div>",
                    unsafe_allow_html=True)
    with mc2:
        st.caption("After Conv1 + ReLU")
        st.image(render_matrix_png(conv1, "relu_out", fig_scale=0.62),
                 use_container_width=True)
    with arr2:
        st.markdown(f"<div style='font-size:2rem;color:{_C['arrow']};"
                    f"padding-top:2.5rem;text-align:center'>→</div>",
                    unsafe_allow_html=True)
    with mc3:
        st.caption("After Conv2 + ReLU")
        st.image(render_matrix_png(conv2, "conv_out", fig_scale=0.62),
                 use_container_width=True)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown('<div class="subsection-title">Receptive Field Visualisation</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="callout">
        Pick any output pixel after the 2nd conv. It depends on a 3×3 region of
        <em>Conv1's output</em>, which itself depends on 3×3 of the original input.
        Combined: a <strong>5×5 receptive field</strong>.
        </div>""", unsafe_allow_html=True)
        rf5 = {(3+dr, 3+dc) for dr in range(-2, 3) for dc in range(-2, 3)
               if 0 <= 3+dr < 8 and 0 <= 3+dc < 8}
        st.caption("Pixels contributing to output[3,3] — brick-red outline")
        st.image(render_matrix_png(STAR_8x8, "input",
                                   highlight_cells=rf5, fig_scale=0.62),
                 use_container_width=True)
    with col2:
        st.markdown('<div class="subsection-title">📊 Parameter Comparison</div>',
                    unsafe_allow_html=True)
        C  = st.slider("Number of Channels (C)", 1, 64, 32, key="step2_C")
        p2 = 2*(9*C*C+C); p5 = 25*C*C+C; sav = (1-p2/p5)*100
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-box metric-box-purple">
            <div class="val">{p2:,}</div>
            <div class="lbl">Two 3×3 Convs (params)</div>
          </div>
          <div class="metric-box metric-box-amber">
            <div class="val">{p5:,}</div>
            <div class="lbl">One 5×5 Conv (params)</div>
          </div>
          <div class="metric-box metric-box-green">
            <div class="val">{sav:.1f}%</div>
            <div class="lbl">Parameter Savings</div>
          </div>
        </div>
        <div class="callout callout-green">
        <strong>Result:</strong> Same receptive field,
        <strong>{sav:.1f}% fewer parameters</strong> at C={C},
        plus two ReLUs instead of one.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# STEP 3 – Three 3×3 = 7×7 RF
# ══════════════════════════════════════════════════════════════════
def story_step3():
    st.markdown("""
    <div class="card card-amber">
    <b>Step 3 of 5:</b> Stack <em>three</em> 3×3 convolutions. Receptive field → 7×7.
    This is VGG's Block 3, 4, and 5 pattern.
    </div>""", unsafe_allow_html=True)

    conv1 = _relu(_apply_conv(STAR_8x8, EDGE_FILTER))
    conv2 = _relu(_apply_conv(conv1,    EDGE_FILTER))
    conv3 = _relu(_apply_conv(conv2,    EDGE_FILTER))

    st.markdown('<div class="subsection-title">Three Conv Passes</div>',
                unsafe_allow_html=True)
    mc1,a1,mc2,a2,mc3,a3,mc4 = st.columns([3,0.35,3,0.35,3,0.35,3])
    pairs = [
        (mc1, STAR_8x8, "input",    "Input"),
        (mc2, conv1,    "relu_out", "After Conv1+ReLU"),
        (mc3, conv2,    "relu_out", "After Conv2+ReLU"),
        (mc4, conv3,    "conv_out", "After Conv3+ReLU"),
    ]
    for col, mat, pal, cap in pairs:
        with col:
            st.caption(cap)
            st.image(render_matrix_png(mat, pal, fig_scale=0.55),
                     use_container_width=True)
    for arr_col in [a1, a2, a3]:
        with arr_col:
            st.markdown(f"<div style='font-size:1.8rem;color:{_C['arrow']};"
                        f"padding-top:2.5rem;text-align:center'>→</div>",
                        unsafe_allow_html=True)

    st.markdown('<div class="subsection-title">7×7 Receptive Field</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        rf7 = {(3+dr, 3+dc) for dr in range(-3, 4) for dc in range(-3, 4)
               if 0 <= 3+dr < 8 and 0 <= 3+dc < 8}
        st.caption("7×7 receptive field of output[3,3]")
        st.image(render_matrix_png(STAR_8x8, "input",
                                   highlight_cells=rf7, fig_scale=0.62),
                 use_container_width=True)
    with col2:
        C  = st.slider("Number of Channels (C)", 1, 64, 32, key="step3_C")
        p3 = 3*(9*C*C+C); p7 = 49*C*C+C; sav = (1-p3/p7)*100
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-box metric-box-purple">
            <div class="val">{p3:,}</div><div class="lbl">Three 3×3 Convs</div>
          </div>
          <div class="metric-box metric-box-amber">
            <div class="val">{p7:,}</div><div class="lbl">One 7×7 Conv</div>
          </div>
        </div>
        <div class="metric-row">
          <div class="metric-box metric-box-green" style="flex:2">
            <div class="val">{sav:.1f}%</div>
            <div class="lbl">Savings with 3 ReLUs vs 1</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="callout callout-purple">
    <strong>VGG Core Insight:</strong> RF grows +2 per 3×3 layer:<br>
    1 conv → 3×3 &nbsp;|&nbsp; 2 convs → 5×5 &nbsp;|&nbsp; 3 convs → 7×7<br><br>
    Blocks 3–5 stack three (VGG-16) or four (VGG-19) 3×3 convolutions.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# STEP 4 – Full VGG Block
# ══════════════════════════════════════════════════════════════════
def story_step4():
    st.markdown("""
    <div class="card card-red">
    <b>Step 4 of 5:</b> A complete VGG block: Conv→Conv→MaxPool, then double channels.
    </div>""", unsafe_allow_html=True)

    conv1 = _relu(_apply_conv(STAR_8x8, EDGE_FILTER))
    conv2 = _relu(_apply_conv(conv1,    BLUR_FILTER))
    pool  = _maxpool2x2(conv2)

    st.markdown('<div class="subsection-title">VGG Block Pipeline</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("After Conv1+ReLU (8×8)")
        st.image(render_matrix_png(conv1, "relu_out", fig_scale=0.62),
                 use_container_width=True)
    with c2:
        st.caption("After Conv2+ReLU (8×8)")
        st.image(render_matrix_png(conv2, "conv_out", fig_scale=0.62),
                 use_container_width=True)
    with c3:
        st.caption("After MaxPool (4×4)")
        st.image(render_matrix_png(pool, "pool_out", fig_scale=0.90),
                 use_container_width=True)

    st.markdown('<div class="subsection-title">▶ MaxPool Animation</div>',
                unsafe_allow_html=True)
    btn_col, info_col = st.columns([1, 3])
    with btn_col:
        play_pool = st.button("▶ Animate MaxPool", key="play_pool",
                              use_container_width=True)
    with info_col:
        st.markdown("""
        <div class="callout" style="margin:0">
        <strong>MaxPool 2×2:</strong> Takes the max from each 2×2 window.
        8×8 → 4×4. Real VGG-16: 224→112→56→28→14→7 after 5 pools.
        </div>""", unsafe_allow_html=True)

    pool_ph = st.empty()
    if play_pool:
        partial_pool = np.zeros((4, 4))
        for pr in range(4):
            for pc in range(4):
                hi = {(2*pr+dr, 2*pc+dc) for dr in range(2) for dc in range(2)}
                partial_pool[pr, pc] = conv2[2*pr:2*pr+2, 2*pc:2*pc+2].max()
                with pool_ph.container():
                    pa, pb = st.columns(2)
                    with pa:
                        st.caption("Conv2 out (pool window highlighted)")
                        st.image(render_matrix_png(conv2, "conv_out",
                                                   highlight_cells=hi,
                                                   fig_scale=0.60),
                                 use_container_width=True)
                    with pb:
                        st.caption("Pool output (building…)")
                        st.image(render_matrix_png(partial_pool, "pool_out",
                                                   fig_scale=0.90),
                                 use_container_width=True)
                time.sleep(0.35)
        pool_ph.empty()

    # ── Channel doubling chips ─────────────────────────────────────
    st.markdown('<div class="subsection-title">📈 Channel Doubling Pattern</div>',
                unsafe_allow_html=True)
    chips = [
        ("Input",   "224×224", "3",    "#5a6472", "#dde2e8"),
        ("Block 1", "112×112", "64",   "#3d5a80", "#d6e4f0"),
        ("Block 2", "56×56",   "128",  "#4a6080", "#d0dce8"),
        ("Block 3", "28×28",   "256",  "#4a7080", "#cce0e8"),
        ("Block 4", "14×14",   "512",  "#4a7060", "#cce4d8"),
        ("Block 5", "7×7",     "512",  "#5a7050", "#d4e2cc"),
        ("FC",      "1×1",     "4096", "#7a5060", "#e8d4da"),
    ]
    inner = ""
    for i, (name, size, ch, fg, bg) in enumerate(chips):
        if i > 0:
            inner += (f'<div style="font-size:1.3rem;color:{_C["arrow"]};'
                      f'padding-top:1.4rem;flex-shrink:0">→</div>')
        inner += f"""
        <div style="background:{bg};color:{fg};border:1.5px solid {fg}55;
                    border-radius:10px;padding:0.55rem 0.8rem;text-align:center;
                    min-width:72px;flex-shrink:0;
                    box-shadow:0 2px 6px rgba(0,0,0,0.08)">
          <div style="font-weight:800;font-size:0.95rem">{ch}ch</div>
          <div style="font-size:0.68rem;opacity:0.85;margin-top:2px">{name}</div>
          <div style="font-size:0.62rem;opacity:0.65;margin-top:1px">{size}</div>
        </div>"""
    st.markdown(
        f'<div style="display:flex;align-items:center;flex-wrap:wrap;'
        f'gap:5px;margin:1rem 0;overflow:visible;padding:0.75rem;'
        f'background:{_C["bg_box"]};border-radius:12px;'
        f'border:1px solid #dde2e8">{inner}</div>',
        unsafe_allow_html=True)

    st.markdown("""
    <div class="callout callout-green">
    <strong>The Trade-off:</strong> Spatial size halves after each MaxPool, channel count
    doubles (up to 512). Total information volume stays roughly constant throughout.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# STEP 5 – Full VGG-16 Architecture
# ══════════════════════════════════════════════════════════════════
def story_step5():
    st.markdown("""
    <div class="card card-blue">
    <b>Step 5 of 5:</b> The complete VGG-16 architecture — 5 conv blocks + 3 FC layers
    + softmax.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="subsection-title">🏗️ VGG-16 Architecture Diagram</div>',
                unsafe_allow_html=True)
    blocks_spec = [
        {"label":"Input",      "sub":"224×224×3",
         "ops":["RGB Image"],
         "color":"#475569","bg":"#f1f5f9"},
        {"label":"Block 1",    "sub":"224→112×64",
         "ops":["Conv 3×3 (64)","ReLU","Conv 3×3 (64)","ReLU","MaxPool 2×2"],
         "color":"#1d4ed8","bg":"#eff6ff"},
        {"label":"Block 2",    "sub":"112→56×128",
         "ops":["Conv 3×3 (128)","ReLU","Conv 3×3 (128)","ReLU","MaxPool 2×2"],
         "color":"#7c3aed","bg":"#faf5ff"},
        {"label":"Block 3",    "sub":"56→28×256",
         "ops":["Conv 3×3 (256) ×3","ReLU ×3","MaxPool 2×2"],
         "color":"#be185d","bg":"#fdf2f8"},
        {"label":"Block 4",    "sub":"28→14×512",
         "ops":["Conv 3×3 (512) ×3","ReLU ×3","MaxPool 2×2"],
         "color":"#b45309","bg":"#fffbeb"},
        {"label":"Block 5",    "sub":"14→7×512",
         "ops":["Conv 3×3 (512) ×3","ReLU ×3","MaxPool 2×2"],
         "color":"#15803d","bg":"#f0fdf4"},
        {"label":"Classifier", "sub":"→1000 classes",
         "ops":["Flatten","FC 4096+ReLU","Dropout 0.5",
                "FC 4096+ReLU","Dropout 0.5","FC 1000","Softmax"],
         "color":"#b91c1c","bg":"#fef2f2"},
    ]
    arch = '<div class="arch-flow">'
    for i, blk in enumerate(blocks_spec):
        if i > 0:
            arch += '<div class="arch-arrow">→</div>'
        ops_html = "".join(
            f'<div class="arch-op" style="color:{blk["color"]}">{op}</div>'
            for op in blk["ops"]
        )
        arch += (f'<div class="arch-box" style="background:{blk["bg"]};'
                 f'border-color:{blk["color"]}">'
                 f'<div class="arch-label" style="color:{blk["color"]}">'
                 f'{blk["label"]}</div>{ops_html}'
                 f'<div class="arch-sub">{blk["sub"]}</div></div>')
    arch += "</div>"
    st.markdown(arch, unsafe_allow_html=True)

    st.markdown('<div class="subsection-title">📋 VGG-16 vs VGG-19 Comparison</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <table class="cmp-table">
      <tr><th>Property</th><th>VGG-16</th><th>VGG-19</th></tr>
      <tr><td>Total Layers</td><td>16 weight layers</td><td>19 weight layers</td></tr>
      <tr><td>Conv Layers</td><td>13 conv layers</td><td>16 conv layers</td></tr>
      <tr><td>Block 3 Convs</td><td>3 convs (3×3)</td><td>4 convs (3×3)</td></tr>
      <tr><td>Block 4 Convs</td><td>3 convs (3×3)</td><td>4 convs (3×3)</td></tr>
      <tr><td>Block 5 Convs</td><td>3 convs (3×3)</td><td>4 convs (3×3)</td></tr>
      <tr><td>Total Parameters</td><td>~138 million</td><td>~143 million</td></tr>
      <tr><td>All Filter Sizes</td><td>3×3 only ✅</td><td>3×3 only ✅</td></tr>
      <tr><td>ImageNet Top-1</td><td>71.6%</td><td>72.4%</td></tr>
      <tr><td>ImageNet Top-5</td><td>90.4%</td><td>90.9%</td></tr>
    </table>""", unsafe_allow_html=True)

    st.markdown(
        '<div class="subsection-title">🏆 Simulated Mini-VGG Classification</div>',
        unsafe_allow_html=True)
    sim_classes = ["Star","Spiral","Cross","Diamond","Circle"]
    sim_probs   = [82, 8, 5, 3, 2]
    bar_colors  = [_C["slate"], _C["teal"], _C["sage"], _C["sand"], _C["rose"]]
    bars = ""
    for cls, prob, color in zip(sim_classes, sim_probs, bar_colors):
        bars += f"""
        <div class="bar-wrap">
          <div class="bar-row-label">
            <span style="color:{_C['text']}">{cls}</span>
            <b style="color:{color}">{prob}%</b>
          </div>
          <div class="bar-bg">
            <div class="bar-fill" style="width:{prob}%;background:{color}"></div>
          </div>
        </div>"""
    st.markdown(
        f'<div class="card card-green" style="max-width:500px">{bars}</div>',
        unsafe_allow_html=True)

    st.markdown("""
    <div class="card card-blue" style="margin-top:1.5rem">
    <h3 style="color:#1d4ed8;margin-bottom:0.8rem">🎓 What VGG Taught the World</h3>
    <ul style="color:#1e293b;line-height:1.9;padding-left:1.2rem">
      <li><strong>Depth beats large filters</strong> — 3×3 stacks outperform large
          filters in accuracy <em>and</em> efficiency.</li>
      <li><strong>Uniform architecture</strong> — only 3×3 convolutions throughout.</li>
      <li><strong>More layers, more abstractions</strong> — edges→textures→objects.</li>
      <li><strong>Lives on</strong> — ResNets and transformers borrow the same insight.
          </li>
    </ul>
    <p style="color:#64748b;margin-bottom:0;font-size:0.85rem;margin-top:0.8rem">
    Simonyan &amp; Zisserman, ICLR 2015.
    </p>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SECTION B WRAPPER
# ══════════════════════════════════════════════════════════════════
def section_vgg_story():
    st.markdown(
        '<div class="section-title">📖 The VGG Story – The Power of 3×3</div>',
        unsafe_allow_html=True)
    if "story_step" not in st.session_state:
        st.session_state.story_step = 0
    step        = st.session_state.story_step
    total_steps = 6
    st.markdown(render_step_indicator(step, total_steps), unsafe_allow_html=True)
    step_titles = [
        "Step 0 – The 8×8 Image",
        "Step 1 – A Single 3×3 Convolution",
        "Step 2 – Two 3×3 = 5×5 Receptive Field",
        "Step 3 – Three 3×3 = 7×7 Receptive Field",
        "Step 4 – Full VGG Block (Conv→Pool→Channels)",
        "Step 5 – VGG-16 Full Architecture & Classification",
    ]
    st.markdown(f"<h3 style='color:#1e3a5f;margin-bottom:0.5rem'>"
                f"{step_titles[step]}</h3>", unsafe_allow_html=True)
    [story_step0, story_step1, story_step2,
     story_step3, story_step4, story_step5][step]()

    st.markdown("<br>", unsafe_allow_html=True)
    nav1, nav2, nav3 = st.columns([1, 3, 1])
    with nav1:
        if step > 0:
            if st.button("◀ Previous", key="prev_step", use_container_width=True):
                st.session_state.story_step -= 1; st.rerun()
    with nav2:
        pct = int((step / (total_steps-1)) * 100)
        st.markdown(f"""
        <div style="text-align:center;color:#64748b;
                    font-size:0.85rem;padding-top:0.5rem">
          Progress: <strong style="color:#1d4ed8">{pct}%</strong>
          &nbsp;—&nbsp; Step {step+1} of {total_steps}
        </div>""", unsafe_allow_html=True)
    with nav3:
        if step < total_steps - 1:
            if st.button("Next ▶", key="next_step", use_container_width=True):
                st.session_state.story_step += 1; st.rerun()
        else:
            if st.button("↺ Restart", key="restart_step", use_container_width=True):
                st.session_state.story_step = 0; st.rerun()


# ──────────────────────────────────────────────
# SIDEBAR  (model selector lives here)
# ──────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1rem 0 0.5rem">
          <div style="font-size:2.8rem">🧠</div>
          <div style="font-size:1.1rem;font-weight:800;color:#7dd3fc">VGG Explorer</div>
          <div style="font-size:0.75rem;color:#94a3b8;margin-top:4px">
            Interactive CNN Education</div>
        </div>
        <hr style="border-color:#334155;margin:0.8rem 0">""",
        unsafe_allow_html=True)

        section = st.radio(
            "Navigate to",
            ["🔬 Feature Map Explorer", "📖 The VGG Story"],
            key="nav_section",
        )

        st.markdown("<hr style='border-color:#334155;margin:0.8rem 0'>",
                    unsafe_allow_html=True)

        # ── Model selector — lazy: only the chosen model is loaded ──
        st.markdown(
            "<div style='font-size:0.82rem;color:#7dd3fc;font-weight:700;"
            "margin-bottom:6px'>Select Model</div>",
            unsafe_allow_html=True)
        selected_model = st.radio(
            "model_radio",
            ["VGG-16", "VGG-19"],
            index=0,
            label_visibility="collapsed",
            key="selected_model",
        )
        st.markdown(
            "<div style='font-size:0.72rem;color:#64748b;margin-top:4px'>"
            "Only the selected model is loaded into memory.</div>",
            unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#334155;margin:0.8rem 0'>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.78rem;color:#94a3b8">
        <b style="color:#7dd3fc">About</b><br><br>
        • Live feature map extraction<br>
        • Model loading (one at a time)<br>
        • Interactive 8×8 matrix demos<br>
        • Receptive field growth<br>
        • Parameter efficiency analysis<br><br>
        <b style="color:#7dd3fc">Reference</b><br>
        Simonyan &amp; Zisserman, ICLR 2015
        </div>""", unsafe_allow_html=True)

    return section, selected_model


# ──────────────────────────────────────────────
# HERO / FOOTER / MAIN
# ──────────────────────────────────────────────
def render_hero():
    st.markdown("""
    <div class="hero-header">
      <h1>🧠 VGG Architecture Explorer</h1>
      <p>Visualize VGG-16 &amp; VGG-19 feature maps in real time &amp; learn why
      <strong style="color:#fde68a">3×3 filters changed deep learning forever</strong>.
      Upload an image or follow the interactive story.</p>
    </div>""", unsafe_allow_html=True)

def render_footer():
    st.markdown("""
    <div class="footer">
      Built with <strong>Streamlit</strong> · <strong>PyTorch</strong> ·
      <strong>torchvision</strong> &nbsp;|&nbsp;
      VGG-16 &amp; VGG-19 pretrained on <strong>ImageNet</strong> &nbsp;|&nbsp;
      <strong>Simonyan &amp; Zisserman, ICLR 2015</strong>
    </div>""", unsafe_allow_html=True)

def main():
    render_hero()
    section, selected_model = render_sidebar()
    if section == "🔬 Feature Map Explorer":
        section_feature_explorer(selected_model)
    else:
        section_vgg_story()
    render_footer()

if __name__ == "__main__":
    main()
