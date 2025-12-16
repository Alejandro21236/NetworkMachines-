#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image-processing domain thermodynamics (frequency space) under Boltzmann-like conditions.

What this does
--------------
- Loads your RT3 model (WSItoRT) and trains exactly ONE epoch with plain SGD
  (fixed learning rate, no momentum, no weight decay) to approximate Langevin/Boltzmann conditions.
- Tracks the patch-embedding convolutional filters (ViT-style) that define how images are decomposed.
- For each training step, computes the 2D Fourier magnitude response of those filters, forms a
  normalized power distribution over frequency, and computes the Boltzmann/Shannon entropy:
      S_ω = -Σ p(f) log p(f)
  (k_B set to 1).
- Optionally, on a small FIXED reference set of patches, computes the spectral entropy of:
    (a) raw input patches and
    (b) their *filtered* responses via the current patch-embedding filters.
- Produces:
    * MP4 video of spectral entropy trajectory and spectrum snapshots over the epoch
    * PNGs of spectra and eigen-summaries
    * JSON with metrics (entropy curves, diffusion proxy, etc.)

Why this is "image-processing domain"
-------------------------------------
We analyze the *filters' optical transfer function* via FFT of the patch-embedding conv weights.
As training progresses, their frequency response re-organizes; the entropy of the spectral power
distribution evolves. This is a direct, causal image-processing view tied to parameters.

Requirements
------------
numpy, torch, matplotlib, imageio. (No ffmpeg needed; uses imageio)

Usage (example)
---------------
python epoch_imgfreq_entropy.py \
  --patches_dir /fs/scratch/PAS2942/TCGA_DS_1/20x/BRCA/patches \
  --labels_csv  /fs/scratch/PAS2942/Alejandro/RT/labels.csv \
  --out_dir     /fs/scratch/PAS2942/Alejandro/RT/outputs/imgfreq_epoch1 \
  --lr 5e-4 --batch_size 1 --max_patches 64 --ref_patches 64 --frameskip 1
"""

import os, math, json, argparse, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# -----------------------------
# Robust RT3 import (no env edits required)
# -----------------------------
import importlib.util
def _load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

HERE = Path(__file__).resolve().parent
RT3 = None
# Try common locations
for cand in [HERE / "RT3.py", HERE / "RT3" / "RT3.py", HERE.parent / "RT3.py", HERE.parent / "RT3" / "RT3.py"]:
    if cand.exists():
        RT3 = _load_mod(cand, "RT3")
        break
if RT3 is None:
    raise FileNotFoundError("Could not locate RT3.py. Put it next to this script or inside ./RT3/RT3.py")

WSItoRT         = RT3.WSItoRT
SlideBagDataset = RT3.SlideBagDataset
ensure_dir      = RT3.ensure_dir
seed_everything = RT3.seed_everything
safe_collate    = RT3.safe_collate
pd              = RT3.pd

# -----------------------------
# Helpers
# -----------------------------
def find_patch_embed_conv(model: nn.Module):
    """
    Try hard to find a ViT-style patch embedding conv:
    - Look for Conv2d with kernel_size == stride > 1 (common in ViT patchifying conv)
    - Prefer a module named like '*patch*' or '*embed*'
    Returns (module, weight_tensor)
    """
    candidates = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            ks = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
            st = m.stride if isinstance(m.stride, tuple) else (m.stride, m.stride)
            if ks == st and ks[0] > 1 and ks[1] > 1:
                score = 0
                lname = name.lower()
                if "patch" in lname: score += 2
                if "embed" in lname or "proj" in lname: score += 1
                candidates.append((score, name, m))
    if not candidates:
        # fallback: any reasonably sized conv; last resort
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size[0] > 1 and m.kernel_size[1] > 1:
                candidates.append((0, name, m))
    if not candidates:
        raise RuntimeError("No suitable patch-embedding conv found.")
    candidates.sort(key=lambda x: (x[0], -x[2].kernel_size[0]*x[2].kernel_size[1]), reverse=True)
    _, best_name, best = candidates[0]
    return best_name, best

def fft2_power_spectrum(w: torch.Tensor, pad: int = 0):
    """
    2D FFT magnitude-squared of filters.
    w: [out_c, in_c, kh, kw] tensor (float32, CPU or CUDA)
    Returns power spectrum averaged over (out_c, in_c): [kh+2p, kw+2p]
    """
    # center weights to remove DC bias per-filter (optional but useful)
    w = w - w.mean(dim=(-2, -1), keepdim=True)
    if pad > 0:
        w = F.pad(w, (pad, pad, pad, pad))  # left,right,top,bottom
    # FFT
    Wf = torch.fft.fft2(w, dim=(-2, -1))
    P = (Wf.real**2 + Wf.imag**2).mean(dim=(0,1))  # average over out,in
    # shift zero-freq to center for plotting
    P = torch.fft.fftshift(P, dim=(-2, -1))
    return P  # [H, W], float32 complex-power averaged

def radial_bins(h, w, nbins=64):
    """
    Precompute integer bin indices for radial averaging on an HxW grid with center at (H/2, W/2).
    """
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r = (r / r.max())  # normalize to [0,1]
    bins = np.clip((r * nbins).astype(int), 0, nbins - 1)
    return bins

def spectral_entropy_from_power(P: np.ndarray, bins: np.ndarray):
    """
    Shannon/Boltzmann-Gibbs entropy over radial frequency bins.
    P: 2D power spectrum (nonnegative)
    bins: integer bin map same shape as P defining frequency rings
    Returns scalar S_ω and normalized histogram p(f).
    """
    P = np.asarray(P, dtype=np.float64)
    P[P < 0] = 0.0
    # Sum power per bin (radial average without dividing by count keeps probability mass)
    nb = bins.max() + 1
    mass = np.bincount(bins.ravel(), weights=P.ravel(), minlength=nb).astype(np.float64)
    if mass.sum() <= 0:
        return float("nan"), np.zeros_like(mass)
    p = mass / mass.sum()
    # S = -Σ p log p (natural log, k_B=1)
    # avoid log(0)
    nz = p[p > 0]
    S = float(-(nz * np.log(nz)).sum())
    return S, p

def running_mean(x):
    out = []
    s = 0.0
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
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_patches", type=int, default=64)
    ap.add_argument("--ref_patches", type=int, default=64, help="fixed reference patches for optional data spectral entropy")
    ap.add_argument("--lr", type=float, default=5e-4, help="fixed LR (Boltzmann-friendly)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frameskip", type=int, default=1)
    ap.add_argument("--nbins", type=int, default=64, help="radial frequency bins")
    ap.add_argument("--pad", type=int, default=8, help="zero-padding around filters before FFT")
    args = ap.parse_args()

    seed_everything(args.seed)
    ensure_dir(args.out_dir)

    # ======= Data =======
    labels = RT3.standardize_columns(pd.read_csv(args.labels_csv))
    labels = RT3.dedupe_columns(labels)
    case_col = RT3.find_case_id_column(labels)
    series = labels[case_col]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    labels["case_id"] = series.astype(str).str.strip().str.lower()
    if "rt_mean" not in labels.columns:
        raise ValueError("labels need 'rt_mean'")
    case_targets_df = labels.groupby("case_id")[["rt_mean"]].mean().reset_index()
    case_targets = {r["case_id"]: r[["rt_mean"]].to_numpy(dtype=np.float32) for _, r in case_targets_df.iterrows()}

    slide_dirs = sorted([d for d in os.listdir(args.patches_dir) if os.path.isdir(os.path.join(args.patches_dir, d))])
    slide_case_ids = [d[:12].lower() for d in slide_dirs]
    keep = [cid in case_targets for cid in slide_case_ids]
    slide_dirs = [s for s, k in zip(slide_dirs, keep) if k]
    slide_case_ids = [c for c, k in zip(slide_case_ids, keep) if k]
    counts = [len(list((Path(args.patches_dir) / s).glob("*.png"))) for s in slide_dirs]
    has_png = [c > 0 for c in counts]
    slide_dirs = [s for s, h in zip(slide_dirs, has_png) if h]
    slide_case_ids = [c for c, h in zip(slide_case_ids, has_png) if h]
    if len(slide_dirs) == 0:
        raise RuntimeError("No slides with patches and labels.")

    # 80/20 split
    n = len(slide_dirs)
    idx = np.arange(n); rng = np.random.default_rng(args.seed); rng.shuffle(idx)
    cut = max(1, int(0.8 * n))
    tr_idx, va_idx = idx[:cut], idx[cut:]
    tr_slides = [slide_dirs[i] for i in tr_idx]
    tr_cases  = [slide_case_ids[i] for i in tr_idx]
    train_ds = SlideBagDataset(tr_slides, tr_cases, case_targets, args.patches_dir, max_patches=args.max_patches, train=True)
    dl_kwargs = dict(num_workers=0, pin_memory=False, collate_fn=safe_collate)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **dl_kwargs)

    # Fixed reference tiny set (for optional data-domain spectral entropy)
    ref_rng = np.random.default_rng(args.seed + 1)
    ref_subset = ref_rng.choice(tr_slides, size=min(len(tr_slides), max(1, args.ref_patches)), replace=False)
    ref_cases = [s[:12].lower() for s in ref_subset]
    ref_ds = SlideBagDataset(list(ref_subset), ref_cases, case_targets, args.patches_dir, max_patches=1, train=False)
    ref_loader = DataLoader(ref_ds, batch_size=8, shuffle=False, **dl_kwargs)

    # ======= Model + Boltzmann-friendly optimizer =======
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WSItoRT(out_dim=1, freeze_backbone_blocks=3).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.0, weight_decay=0.0)
    loss_fn = nn.SmoothL1Loss(beta=1.0)

    # Locate patch-embedding conv
    name_pe, pe_conv = find_patch_embed_conv(model)
    print(f"[info] using patch-embed conv: {name_pe} | ks={pe_conv.kernel_size} stride={pe_conv.stride}")

    # Precompute radial bins for the FILTER spectrum size (including padding)
    with torch.no_grad():
        kh, kw = pe_conv.weight.shape[-2:]
        H = kh + 2 * args.pad
        W = kw + 2 * args.pad
    rbins = radial_bins(H, W, nbins=args.nbins)

    # Storage
    S_filter_hist, loss_hist, G_trace = [], [], []
    P_filter_last = None
    checkpoints = set()
    ref_S_input, ref_S_output, ref_steps = [], [], []

    # ======= One epoch =======
    model.train()
    step = 0
    for batch in train_loader:
        if batch is None or (isinstance(batch, (list, tuple)) and batch[0] is None):
            continue
        X, y, _, _ = batch
        X = X.to(device); y = y.to(device).view(-1)

        # Forward/backward
        opt.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            y_hat, _, _ = model(X)
            loss = loss_fn(y_hat, y)
        loss.backward()

        # Gradient trace proxy
        g_norm2 = 0.0
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                g_norm2 += float((g * g).mean().item())
        G_trace.append(g_norm2)

        # BEFORE step: current FILTER spectral entropy
        with torch.no_grad():
            w = pe_conv.weight.detach()                                 # [out_c, in_c, kh, kw]
            P = fft2_power_spectrum(w, pad=args.pad)                    # torch [H, W]
            P_np = P.float().cpu().numpy()
            S_filter, _ = spectral_entropy_from_power(P_np, rbins)      # <-- correct call
            S_filter_hist.append(S_filter)
            P_filter_last = P_np

        # Optional checkpoints for reference spectral entropies
        if step % max(1, 20 // max(1, args.frameskip)) == 0:
            with torch.no_grad():
                s_input_vals, s_output_vals = [], []
                c_batches = 0
                for RX, _, _, _ in ref_loader:
                    RX = RX.to(device)

                    # ---- Input spectral entropy (per-batch) ----
                    RXf = torch.fft.fft2(RX, dim=(-2, -1))
                    Pin = (RXf.real**2 + RXf.imag**2).mean(dim=1)       # [B, H, W]
                    Pin = torch.fft.fftshift(Pin, dim=(-2, -1))
                    H_in, W_in = Pin.shape[-2], Pin.shape[-1]
                    rb_in = radial_bins(H_in, W_in, nbins=args.nbins)    # bins that MATCH Pin
                    tmp = []
                    for b in range(Pin.shape[0]):
                        s_in, _ = spectral_entropy_from_power(Pin[b].float().cpu().numpy(), rb_in)
                        if not math.isnan(s_in):
                            tmp.append(s_in)
                    if tmp:
                        s_input_vals.append(float(np.mean(tmp)))

                    # ---- Filtered-output spectral entropy ----
                    Y = F.conv2d(RX, pe_conv.weight, bias=None, stride=pe_conv.stride, padding=0)  # [B, C', H', W']
                    Yf = torch.fft.fft2(Y, dim=(-2, -1))
                    Pout = (Yf.real**2 + Yf.imag**2).mean(dim=1)         # [B, H', W']
                    Pout = torch.fft.fftshift(Pout, dim=(-2, -1))
                    H2, W2 = Pout.shape[-2], Pout.shape[-1]
                    rb2 = radial_bins(H2, W2, nbins=args.nbins)
                    tmp2 = []
                    for b in range(Pout.shape[0]):
                        s_out, _ = spectral_entropy_from_power(Pout[b].float().cpu().numpy(), rb2)
                        if not math.isnan(s_out):
                            tmp2.append(s_out)
                    if tmp2:
                        s_output_vals.append(float(np.mean(tmp2)))

                    c_batches += 1
                    if c_batches >= 2:
                        break

                ref_S_input.append(float(np.mean(s_input_vals)) if s_input_vals else float("nan"))
                ref_S_output.append(float(np.mean(s_output_vals)) if s_output_vals else float("nan"))
                ref_steps.append(int(step))
                checkpoints.add(step)

        # Step
        opt.step()

        loss_hist.append(float(loss.item()))
        step += 1

    # ======= Outputs =======
    out_dir = Path(args.out_dir); ensure_dir(out_dir)
    ep_dir = out_dir / "epoch_001_imgfreq"; ensure_dir(ep_dir)
    (ep_dir / "frames").mkdir(parents=True, exist_ok=True)

    # Running means
    S_filter_rm = running_mean(S_filter_hist)
    G_trace_rm  = running_mean(G_trace)

    # Video frames
    frames = []
    for t in range(0, len(S_filter_hist), max(1, args.frameskip)):
        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.2, 1.2], height_ratios=[1,1], wspace=0.35, hspace=0.45)

        ax1 = fig.add_subplot(gs[0,0]); ax1.plot(S_filter_rm[:t+1], lw=2)
        ax1.set_title("Filter spectral entropy S_ω (running)"); ax1.set_xlabel("step"); ax1.set_ylabel("S_ω")

        ax2 = fig.add_subplot(gs[0,1]); ax2.plot(loss_hist[:t+1], lw=2)
        ax2.set_title("Training loss (SmoothL1)"); ax2.set_xlabel("step"); ax2.set_ylabel("loss")

        ax3 = fig.add_subplot(gs[0,2]); ax3.plot(G_trace_rm[:t+1], lw=2)
        ax3.set_title("Gradient trace (running mean)"); ax3.set_xlabel("step"); ax3.set_ylabel("⟨‖g‖²⟩")

        ax4 = fig.add_subplot(gs[1,0])
        if P_filter_last is not None:
            im = ax4.imshow(np.log1p(P_filter_last), cmap="viridis")
            ax4.set_title("Filter spectrum (log power)")
            fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        else:
            ax4.text(0.5, 0.5, "n/a", ha="center", va="center"); ax4.set_axis_off()

        ax5 = fig.add_subplot(gs[1,1])
        if ref_steps:
            ax5.plot(ref_steps, ref_S_input, marker="o", lw=1.5)
            ax5.set_title("Ref input spectral entropy"); ax5.set_xlabel("step"); ax5.set_ylabel("S_ω(input)")

        ax6 = fig.add_subplot(gs[1,2])
        if ref_steps:
            ax6.plot(ref_steps, ref_S_output, marker="o", lw=1.5)
            ax6.set_title("Ref filtered-output spectral entropy"); ax6.set_xlabel("step"); ax6.set_ylabel("S_ω(output)")

        fig.suptitle(f"Boltzmann-like training: SGD lr={args.lr}, batch={args.batch_size}, no momentum/decay", fontsize=10)
        fpath = ep_dir / "frames" / f"frame_{t:04d}.png"
        fig.tight_layout(); fig.savefig(fpath, dpi=180); plt.close(fig)
        frames.append(imageio.imread(fpath))

    mp4_path = str(ep_dir / "imgfreq_epoch1.mp4")
    imageio.mimsave(mp4_path, frames, fps=max(2, 30 // max(1, args.frameskip)))

    # Final static plots
    plt.figure(figsize=(5,4))
    plt.plot(S_filter_hist, alpha=0.3, label="raw")
    plt.plot(S_filter_rm, lw=2, label="running")
    plt.xlabel("step"); plt.ylabel("S_ω (filters)")
    plt.title("Filter spectral entropy over epoch")
    plt.legend(); plt.tight_layout()
    plt.savefig(str(ep_dir / "S_filter_epoch.png"), dpi=220); plt.close()

    if ref_steps:
        plt.figure(figsize=(5,4))
        plt.plot(ref_steps, ref_S_input, marker="o", label="input")
        plt.plot(ref_steps, ref_S_output, marker="o", label="filtered")
        plt.xlabel("step"); plt.ylabel("S_ω")
        plt.title("Reference spectral entropies")
        plt.legend(); plt.tight_layout()
        plt.savefig(str(ep_dir / "S_ref_epoch.png"), dpi=220); plt.close()

    # Metrics JSON
    out = {
        "patch_embed_name": name_pe,
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "nbins": int(args.nbins),
        "pad": int(args.pad),
        "num_steps": int(len(S_filter_hist)),
        "S_filter_hist": [float(v) for v in S_filter_hist],
        "S_filter_running_last": float(S_filter_rm[-1]) if len(S_filter_rm) else float("nan"),
        "loss_last": float(loss_hist[-1]) if len(loss_hist) else float("nan"),
        "grad_trace_running_last": float(G_trace_rm[-1]) if len(G_trace_rm) else float("nan"),
        "ref_steps": ref_steps,
        "ref_S_input": [float(v) for v in ref_S_input],
        "ref_S_output": [float(v) for v in ref_S_output],
    }
    (ep_dir / "metrics_imgfreq_epoch.json").write_text(json.dumps(out, indent=2))

    print(f"[done] video   → {mp4_path}")
    print(f"[done] plots   → {ep_dir}")
    print(f"[done] metrics → {ep_dir/'metrics_imgfreq_epoch.json'}")
if __name__ == "__main__":
    main()