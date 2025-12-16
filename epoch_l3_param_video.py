#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter-space (weights) intra-epoch dynamics for Layer 3 with Boltzmann conditions.

- Uses RT3.WSItoRT (your model).
- Trains exactly ONE epoch with plain SGD (no momentum, fixed LR, no schedule).
- Logs per-batch gradients and weights for ViT backbone block 3 ("layer 3").
- Estimates gradient covariance Σ, effective diffusion D = (η / (2S)) Σ.
- Computes empirical weight covariance C_emp across the epoch.
- Reports Gaussian differential entropy h = 0.5*log((2πe)^d det C_emp) (low-rank).
- Produces an MP4 video of the layer-3 weight trajectory projected onto top PCs.
- Emits diagnostics to check intra-epoch stationarity of gradient noise.

Requirements: numpy, torch, matplotlib, imageio (pure-Python, no ffmpeg needed).
"""

import os, math, json, argparse, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast  # used only for forward speed (no scaler here)
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# ---- import your hardened code ----
# ---- import your hardened code ----
import importlib, importlib.util
from importlib import import_module
from pathlib import Path
HERE = Path(__file__).resolve().parent

def _load_from_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return None

# Try standard imports first (works if PYTHONPATH points to WORKDIR)
RT3 = None
for cand in ("RT3", "RT3.RT3"):
    try:
        RT3 = import_module(cand)
        break
    except Exception:
        pass

# Fallbacks for common layouts
if RT3 is None:
    candidates = [
        HERE / "RT3.py",
        HERE / "RT3" / "RT3.py",
        HERE.parent / "RT3.py",
        HERE.parent / "RT3" / "RT3.py",
    ]
    for p in candidates:
        if p.exists():
            RT3 = _load_from_path(p, "RT3")
            if RT3 is not None:
                break

if RT3 is None:
    raise FileNotFoundError(
        "Could not locate RT3. Ensure either (a) import RT3 works via PYTHONPATH, "
        "or (b) RT3.py exists at machine/RT3.py or machine/RT3/RT3.py"
    )

WSItoRT         = RT3.WSItoRT
SlideBagDataset = RT3.SlideBagDataset
ensure_dir      = RT3.ensure_dir
seed_everything = RT3.seed_everything
safe_collate    = RT3.safe_collate

# -----------------------------
# Utilities
# -----------------------------
def flatten_params(named_params):
    """Concatenate a list of tensors (same device/dtype) into a 1D vector and keep views for unflatten."""
    flats = [p.reshape(-1) for p in named_params]
    sizes = [f.numel() for f in flats]
    idxs  = np.cumsum([0] + sizes)
    vec   = torch.cat(flats, dim=0)
    def unflatten(vec_like):
        parts = []
        off = 0
        for p in named_params:
            n = p.numel()
            parts.append(vec_like[off:off+n].view_as(p))
            off += n
        return parts
    return vec, sizes, idxs, unflatten

def select_layer3_params(model: nn.Module):
    """
    Grab parameters for ViT backbone block index 3 (0-based).
    Falls back to a heuristic if blocks are not present.
    """
    bb = getattr(model, "backbone", model)
    if hasattr(bb, "blocks"):
        blocks = list(bb.blocks)
        k = 3 if len(blocks) > 3 else (len(blocks) - 1)
        # collect ONLY weights (and biases) inside this block
        params = []
        names  = []
        for name, p in blocks[k].named_parameters(recurse=True):
            if p.requires_grad:
                params.append(p)
                names.append(f"backbone.blocks.{k}.{name}")
        return params, names, k
    # Fallback: take the 4th leaf module with parameters
    counted = 0
    taken = []
    names = []
    for name, mod in bb.named_modules():
        if any(True for _ in mod.parameters(recurse=False)):
            if counted == 3:
                for n, p in mod.named_parameters(recurse=True):
                    if p.requires_grad:
                        taken.append(p); names.append(f"backbone.{name}.{n}")
                return taken, names, counted
            counted += 1
    raise RuntimeError("Could not locate layer 3 parameters.")

def lowrank_logdet_from_cov(C_eigs, d, eps=1e-10, rank_cap=128):
    """
    C_eigs: top-k eigenvalues of covariance (numpy array, nonnegative)
    d: full dimensionality
    Return logdet(C) ≈ sum(log(lambda_i)) + (d-k)*log(eps)
    """
    lam = np.asarray(C_eigs, dtype=np.float64)
    lam = lam[lam > 0]
    k = min(rank_cap, lam.size)
    lam = np.sort(lam)[::-1][:k]
    # ridge for the discarded subspace
    ridge = eps
    logdet = float(np.sum(np.log(lam + 1e-24)) + (d - k) * math.log(ridge))
    return logdet, k

def gaussian_diff_entropy_from_cov_eigs(C_eigs, d, eps=1e-10, rank_cap=128):
    # h = 0.5 * log( (2πe)^d det C )
    logdet, k = lowrank_logdet_from_cov(C_eigs, d, eps=eps, rank_cap=rank_cap)
    h = 0.5 * (d * math.log(2 * math.pi * math.e) + logdet)
    return float(h), k

def running_mean(x):
    s = 0.0
    out = []
    for i, v in enumerate(x, 1):
        s += float(v)
        out.append(s / i)
    return np.array(out, dtype=np.float64)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches_dir", required=True, type=str)
    ap.add_argument("--labels_csv", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--batch_size", type=int, default=1, help="slides per batch; keep 1")
    ap.add_argument("--max_patches", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4, help="fixed LR within the epoch")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epoch_index", type=int, default=1, help="which epoch to record (we will run exactly one)")
    ap.add_argument("--rank_cap", type=int, default=128, help="top-k eigen spectrum for entropy/logdet")
    ap.add_argument("--frameskip", type=int, default=1, help="write every Nth batch as a frame")
    args = ap.parse_args()

    seed_everything(args.seed)
    ensure_dir(args.out_dir)

    # ============ Data ============
    labels = RT3.standardize_columns(RT3.pd.read_csv(args.labels_csv))
    labels = RT3.dedupe_columns(labels)
    case_col = RT3.find_case_id_column(labels)
    series = labels[case_col]
    if isinstance(series, RT3.pd.DataFrame): series = series.iloc[:,0]
    labels["case_id"] = series.astype(str).str.strip().str.lower()
    # single target (rt_mean) to match your study design
    if "rt_mean" not in labels.columns:
        raise ValueError("labels need 'rt_mean'")
    case_targets_df = labels.groupby("case_id")[["rt_mean"]].mean().reset_index()
    case_targets = {r["case_id"]: r[["rt_mean"]].to_numpy(dtype=np.float32) for _, r in case_targets_df.iterrows()}

    slide_dirs = sorted([d for d in os.listdir(args.patches_dir) if os.path.isdir(os.path.join(args.patches_dir, d))])
    slide_case_ids = [d[:12].lower() for d in slide_dirs]
    keep = [cid in case_targets for cid in slide_case_ids]
    slide_dirs = [s for s, k in zip(slide_dirs, keep) if k]
    slide_case_ids = [c for c, k in zip(slide_case_ids, keep) if k]
    counts = [len(list((Path(args.patches_dir)/s).glob("*.png"))) for s in slide_dirs]
    has_png = [c > 0 for c in counts]
    slide_dirs = [s for s, h in zip(slide_dirs, has_png) if h]
    slide_case_ids = [c for c, h in zip(slide_case_ids, has_png) if h]
    if len(slide_dirs) == 0:
        raise RuntimeError("No slides with patches and labels.")

    # simple 80/20 split
    n = len(slide_dirs)
    idx = np.arange(n); rng = np.random.default_rng(args.seed); rng.shuffle(idx)
    cut = max(1, int(0.8 * n))
    tr_idx, va_idx = idx[:cut], idx[cut:]
    tr_slides = [slide_dirs[i] for i in tr_idx]
    va_slides = [slide_dirs[i] for i in va_idx]
    tr_cases  = [slide_case_ids[i] for i in tr_idx]
    va_cases  = [slide_case_ids[i] for i in va_idx]

    train_ds = SlideBagDataset(tr_slides, tr_cases, case_targets, args.patches_dir, max_patches=args.max_patches, train=True)
    valid_ds = SlideBagDataset(va_slides, va_cases, case_targets, args.patches_dir, max_patches=args.max_patches, train=False)
    dl_kwargs = dict(num_workers=0, pin_memory=False, collate_fn=safe_collate)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **dl_kwargs)
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False, **dl_kwargs)

    # ============ Model ============
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WSItoRT(out_dim=1, freeze_backbone_blocks=3).to(device)

    # >>> Boltzmann-friendly optimizer: plain SGD, fixed LR, no momentum, no weight decay
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.0, weight_decay=0.0)
    loss_fn = nn.SmoothL1Loss(beta=1.0)

    # choose layer 3 params
    layer_params, layer_param_names, layer_index = select_layer3_params(model)
    print(f"[info] tracking layer index={layer_index} with {len(layer_params)} tensors")

    # Flatten view helpers for layer 3
    with torch.no_grad():
        w0_vec, _sizes, _idxs, _unflatten = flatten_params(layer_params)
        d = int(w0_vec.numel())

    # Storage
    G_list = []     # gradients per batch (numpy)
    W_list = []     # weights (flattened) per step (numpy)
    loss_hist = []
    grad_trace_hist = []  # trace(Σ_t) running (per-batch centered over a window later)
    steps = 0

    # ============ One epoch with logging ============
    model.train()
    for batch in train_loader:
        if batch is None or (isinstance(batch, (list,tuple)) and batch[0] is None):
            continue
        X, y, _, _ = batch
        X = X.to(device); y = y.to(device).view(-1)

        # zero, forward, loss
        opt.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            y_hat, _, _ = model(X)
            loss = loss_fn(y_hat, y)

        # backward: record gradients for layer 3
        loss.backward()

        # flatten gradient for selected params
        g_parts = []
        for p in layer_params:
            if p.grad is None:
                g_parts.append(torch.zeros_like(p).reshape(-1))
            else:
                g_parts.append(p.grad.detach().reshape(-1))
        g_vec = torch.cat(g_parts, dim=0).detach().float().cpu().numpy()
        G_list.append(g_vec)

        # record current weights before step
        with torch.no_grad():
            w_vec = torch.cat([p.detach().reshape(-1) for p in layer_params], dim=0).float().cpu().numpy()
        W_list.append(w_vec)

        # optimizer step (fixed LR)
        opt.step()

        loss_hist.append(float(loss.item()))
        steps += 1

    # After the epoch, record final weights too (for last increment)
    with torch.no_grad():
        w_vec = torch.cat([p.detach().reshape(-1) for p in layer_params], dim=0).float().cpu().numpy()
    W_list.append(w_vec)

    G = np.stack(G_list, axis=0)  # [B, d]
    W = np.stack(W_list, axis=0)  # [B+1, d]

    # Intra-epoch gradient covariance (centered across batches)
    g_mean = G.mean(axis=0, keepdims=True)
    Gc = G - g_mean
    # trace Σ_t proxy per batch (squared norm) – then average
    grad_trace_hist = np.sum(Gc**2, axis=1) / max(1, Gc.shape[1])
    grad_trace_running = running_mean(grad_trace_hist)

    # Sample covariance Σ (Bessel correction)
    B = Gc.shape[0]
    Sigma = (Gc.T @ Gc) / max(1, (B - 1))  # [d, d] (may be huge)
    # top-k eigen-spectrum of Σ (for diagnostics / diffusion anisotropy)
    # use randomized SVD/eig on covariance via economy SVD on Gc
    # Gc = U S V^T, then Σ = V (S^2/(B-1)) V^T
    U, Svals, Vt = np.linalg.svd(Gc, full_matrices=False)
    sigma_eigs = (Svals**2) / max(1, (B - 1))  # length = min(B, d)
    # Effective diffusion tensor scaling: D = (eta / (2S)) Σ
    eta = float(args.lr)
    S_eff = int(args.batch_size)  # effective batch size in your loader
    D_scale = eta / max(1.0, 2.0 * float(S_eff))  # scalar; diffusion eigs = D_scale * sigma_eigs

    # Empirical weight covariance across the epoch (centered)
    # Use the B+1 snapshots of weights
    Wc = W - W.mean(axis=0, keepdims=True)
    # Economy SVD on Wc to get covariance eigenvalues
    Uw, Sw, Vtw = np.linalg.svd(Wc, full_matrices=False)
    C_eigs = (Sw**2) / max(1, (Wc.shape[0] - 1))  # eigenvalues of covariance in the Vtw basis

    # Differential entropy (Gaussian) with low-rank approximation
    d_full = d
    h_gauss, used_rank = gaussian_diff_entropy_from_cov_eigs(
        C_eigs=C_eigs, d=d_full, eps=1e-8, rank_cap=args.rank_cap
    )

    # ============ Plots & Video ============
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    ep_dir = out_dir / f"epoch_{int(args.epoch_index):03d}_layer3_param"
    ensure_dir(ep_dir)
    (ep_dir / "frames").mkdir(parents=True, exist_ok=True)

    # 2D trajectory in PC space of G (gradient PCs define principal diffusion axes)
    # project W_t onto top-2 right singular vectors of Gc
    V = Vt.T  # [d, r]
    r = V.shape[1]
    k2 = min(2, r)
    P2 = V[:, :k2] if k2 >= 1 else np.eye(d_full, 2)[:,:k2]
    XY = W @ P2  # [B+1, 2]

    # frame rendering
    frames = []
    for t in range(0, W.shape[0], max(1, args.frameskip)):
        fig = plt.figure(figsize=(8, 5))
        gs = fig.add_gridspec(2, 2, width_ratios=[2,1], height_ratios=[1,1], wspace=0.25, hspace=0.35)

        # (1) trajectory up to time t
        ax1 = fig.add_subplot(gs[:,0])
        if k2 >= 2:
            ax1.plot(XY[:t+1,0], XY[:t+1,1], '-', alpha=0.6)
            ax1.scatter(XY[0,0], XY[0,1], s=50, c='green', label='start')
            ax1.scatter(XY[t,0], XY[t,1], s=50, c='red', label='current')
            ax1.set_xlabel("PC1 (grad space)"); ax1.set_ylabel("PC2 (grad space)")
        elif k2 == 1:
            ax1.plot(np.arange(t+1), XY[:t+1,0], '-', alpha=0.6)
            ax1.set_xlabel("step"); ax1.set_ylabel("PC1 (grad space)")
        ax1.set_title(f"Layer 3 weight trajectory (t={t}/{W.shape[0]-1})")
        ax1.legend(loc='best')

        # (2) running mean of grad trace (stationarity proxy)
        ax2 = fig.add_subplot(gs[0,1])
        ax2.plot(grad_trace_running[:max(1,t)], lw=2)
        ax2.set_title("Intra-epoch gradient trace (running mean)")
        ax2.set_xlabel("batch"); ax2.set_ylabel("trace(Σ)/d")

        # (3) entropy so far (recompute on prefix for animation)
        # prefix covariance eigenvalues via economy SVD on W[:t+1]
        if t+1 >= 3:
            Wp = W[:t+1]
            Wpc = Wp - Wp.mean(axis=0, keepdims=True)
            _, Swp, _ = np.linalg.svd(Wpc, full_matrices=False)
            Ceigs_p = (Swp**2) / max(1, (Wpc.shape[0]-1))
            h_p, _ = gaussian_diff_entropy_from_cov_eigs(Ceigs_p, d_full, eps=1e-8, rank_cap=args.rank_cap)
        else:
            h_p = float('nan')

        ax3 = fig.add_subplot(gs[1,1])
        ax3.plot(np.arange(len(loss_hist)), loss_hist, lw=2)
        ax3.set_title(f"Train loss (this epoch) | h_gauss(final)={h_gauss:.3e}")
        ax3.set_xlabel("batch"); ax3.set_ylabel("SmoothL1 loss")

        fig.suptitle(f"Boltzmann-like conditions: SGD fixed lr={args.lr}, batch_size={args.batch_size}, no momentum/decay", fontsize=10)
        fpath = ep_dir / "frames" / f"frame_{t:04d}.png"
        fig.tight_layout()
        fig.savefig(fpath, dpi=180)
        plt.close(fig)
        frames.append(imageio.imread(fpath))

    # write video
    mp4_path = str(ep_dir / "layer3_intraepoch_weights.mp4")
    imageio.mimsave(mp4_path, frames, fps=max(2, 30 // max(1, args.frameskip)))

    # Σ spectrum and D scale plot
    plt.figure(figsize=(5,4))
    s = np.sort(sigma_eigs)[::-1][:min(64, sigma_eigs.size)]
    plt.plot(np.arange(1, len(s)+1), s, lw=2)
    plt.yscale("log"); plt.xlabel("eigen-index"); plt.ylabel("eigenvalue")
    plt.title("Top Σ eigenvalues (layer 3 grads)")
    plt.tight_layout()
    plt.savefig(str(ep_dir / "sigma_spectrum.png"), dpi=220)
    plt.close()

    # Save diagnostics/metrics
    out = {
        "layer_index": int(layer_index),
        "num_params_tracked": int(d_full),
        "num_batches": int(B),
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "entropy_gaussian_final": float(h_gauss),
        "rank_used_for_entropy": int(used_rank),
        "D_scale": float(D_scale),  # diffusion scalar that multiplies Σ
        "grad_trace_running_last": float(grad_trace_running[-1]) if grad_trace_running.size else float('nan')
    }
    (ep_dir / "metrics_param_epoch.json").write_text(json.dumps(out, indent=2))

    print(f"[done] frames → {ep_dir/'frames'}")
    print(f"[done] video  → {mp4_path}")
    print(f"[done] Σ spectrum → {ep_dir/'sigma_spectrum.png'}")
    print(f"[done] metrics → {ep_dir/'metrics_param_epoch.json'}")

if __name__ == "__main__":
    main()
