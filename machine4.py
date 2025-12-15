#!/usr/bin/env python3
# Memory-lean machine3_best.py (conserves all functionalities, adds stability + OT + 3D)

import os
# ---- tame thread counts (pre-import) ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys, importlib.util, pathlib, json, math, argparse, random, hashlib, gc
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
torch.set_num_threads(1)  # keep BLAS threads low
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import networkx as nx
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score  # (kept; ΔAUC placeholder uses None)

# -----------------------------
# RT3 resolver (loads machine/RT3/RT3.py first, then machine/RT3.py, no parent crawl)
# -----------------------------
def _load_module_from(file_path: pathlib.Path, name: str = "RT3"):
    spec = importlib.util.spec_from_file_location(name, str(file_path))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def load_rt3(explicit_path: str | None = None):
    candidates: list[pathlib.Path] = []
    if explicit_path:
        p = pathlib.Path(explicit_path)
        candidates.append(p if p.suffix == ".py" else (p / "RT3.py"))
    here = pathlib.Path(__file__).resolve().parent
    candidates.append(here / "RT3" / "RT3.py")  # prefer subfolder copy if present
    candidates.append(here / "RT3.py")          # fallback to local file
    rt_root = os.environ.get("RT_ROOT")
    if rt_root:
        candidates.append(pathlib.Path(rt_root) / "RT3.py")
    for sp in list(sys.path):
        try:
            candidates.append(pathlib.Path(sp) / "RT3.py")
        except Exception:
            pass
    tried = []
    for path in candidates:
        if not path:
            continue
        tried.append(str(path))
        if path.is_file():
            mod = _load_module_from(path, "RT3")
            if mod is not None and hasattr(mod, "WSItoRT"):
                print(f"[RT3] using file: {path}")
                return mod
    try:
        import RT3 as mod  # type: ignore
        if hasattr(mod, "WSItoRT"):
            print(f"[RT3] using module from sys.path: {getattr(mod,'__file__','<namespace>')}")
            return mod
        raise ImportError(f"[RT3] Imported {getattr(mod,'__file__',None)} but lacks WSItoRT")
    except Exception as e:
        raise ImportError("Failed to import RT3. Tried:\n  " + "\n  ".join(tried)) from e

RT3 = load_rt3(os.environ.get("RT3_PATH"))
# export RT3 symbols
WSItoRT        = RT3.WSItoRT
SlideBagDataset= RT3.SlideBagDataset
safe_collate   = RT3.safe_collate
ensure_dir     = RT3.ensure_dir
seed_everything= RT3.seed_everything

# -----------------------------
# Helpers (unchanged behavior)
# -----------------------------
def zscore_cols(X):
    m = X.mean(axis=0, keepdims=True)
    s = X.std(axis=0, keepdims=True) + 1e-8
    return (X - m) / s

def topk_symmetric(W, k):
    d = W.shape[0]
    A = np.zeros_like(W, dtype=float)
    for i in range(d):
        idx = np.argsort(-np.abs(W[i]))[:k+1]
        for j in idx:
            if i == j: continue
            A[i,j] = W[i,j]
            if abs(W[j,i]) < abs(W[i,j]): A[j,i] = W[i,j]
    np.fill_diagonal(A, 0.0)
    return A

def build_graph_from_activations(A, topk=15):
    A = np.asarray(A, dtype=float)
    A = zscore_cols(A)
    C = np.corrcoef(A, rowvar=False)
    C[np.isnan(C)] = 0.0
    C = np.clip(C, -1.0, 1.0)
    S = topk_symmetric(C, k=topk)
    G = nx.Graph()
    d = S.shape[0]
    G.add_nodes_from(range(d))
    nz = np.where(np.abs(S) > 0)
    for i, j in zip(nz[0], nz[1]):
        if i < j:
            G.add_edge(int(i), int(j), weight=float(S[i, j]))
    return G, S

def greedy_modules(G):
    if G.number_of_edges() == 0: return []
    try:
        comms = nx.algorithms.community.greedy_modularity_communities(G, weight="weight")
    except Exception:
        comms = []
    mods = []
    for k, c in enumerate(comms):
        nodes = sorted(list(c))
        if len(nodes) >= 2: mods.append((f"module_{k}", nodes))
    return mods
# ====== BASLINE METRICS: SVCCA / PWCCA / CKA ======
def _center(X):
    X = np.asarray(X, dtype=np.float32)
    X -= X.mean(axis=0, keepdims=True)
    return X

def _svd_trim(X, var_keep=0.99, max_dim=128):
    # low-rank SVD keeping 99% variance, clipped to max_dim
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    lam = (S**2)
    cum = np.cumsum(lam) / (lam.sum() + 1e-12)
    r = int(np.searchsorted(cum, var_keep) + 1)
    r = int(max(1, min(r, max_dim, Vt.shape[0])))
    return U[:, :r], S[:r], Vt[:r, :]

def _canonical_corrs(X, Y, eps=1e-6):
    # CCA via SVD on whitened cross-covariance (Raghu et al., SVCCA)
    X = _center(X); Y = _center(Y)
    # tiny guard if degenerate
    if X.shape[0] < 2 or Y.shape[0] < 2 or X.shape[1] < 1 or Y.shape[1] < 1:
        return np.array([0.0], dtype=np.float32)

    # whiten
    Cxx = (X.T @ X) / (X.shape[0] - 1) + eps*np.eye(X.shape[1], dtype=np.float32)
    Cyy = (Y.T @ Y) / (Y.shape[0] - 1) + eps*np.eye(Y.shape[1], dtype=np.float32)
    Cxy = (X.T @ Y) / (X.shape[0] - 1)

    # invert sqrt via eigendecomp
    Sx, Ux = np.linalg.eigh(Cxx)
    Sy, Uy = np.linalg.eigh(Cyy)
    Sx = np.clip(Sx, eps, None); Sy = np.clip(Sy, eps, None)
    invsqrt_Cxx = (Ux @ (np.diag(1.0/np.sqrt(Sx)) @ Ux.T)).astype(np.float32)
    invsqrt_Cyy = (Uy @ (np.diag(1.0/np.sqrt(Sy)) @ Uy.T)).astype(np.float32)

    T = invsqrt_Cxx @ Cxy @ invsqrt_Cyy
    # singular values of T are canonical correlations
    s = np.linalg.svd(T, compute_uv=False)
    s = np.clip(s, 0.0, 1.0).astype(np.float32)
    return s

def svcca_pwcca(X_prev, X_cur, var_keep=0.99, max_dim=128):
    # project each into its SV basis (variance-trim), then CCA
    Xp = _center(X_prev); Yp = _center(X_cur)
    # compress features (columns)
    _, _, Vx = _svd_trim(Xp, var_keep=var_keep, max_dim=max_dim)
    _, _, Vy = _svd_trim(Yp, var_keep=var_keep, max_dim=max_dim)
    Xc = Xp @ Vx.T
    Yc = Yp @ Vy.T
    corr = _canonical_corrs(Xc, Yc)           # vector of canonical corrs
    svcca = float(np.mean(corr))

    # PWCCA weights: projection of mean-removed activations onto CCA directions
    # As an efficient proxy, weight by variance explained of each canonical dim in Xc
    if corr.size == 0:
        return svcca, float("nan")
    # variance weights (sum to 1)
    vx = np.var(Xc, axis=0).astype(np.float32)
    if vx.sum() <= 0:
        pwcca = svcca
    else:
        wx = vx / (vx.sum() + 1e-12)
        k = min(wx.size, corr.size)
        pwcca = float((wx[:k] * corr[:k]).sum())
    return svcca, pwcca

def continuity_hungarian(prev_mods, cur_mods, prev_S, cur_S):
    """
    1-1 continuity (Hungarian) without OT; returns:
    - jaccard turnover (1 - J) averaged over matches
    - matched list [(prev, curr, sim)]
    """
    if not prev_mods or not cur_mods:
        return float("nan"), []

    U = module_centroids(prev_S, prev_mods)
    V = module_centroids(cur_S, cur_mods)
    S_sim = cosine_sim_matrix(U, V)
    cost = 1.0 - S_sim
    ri, ci = linear_sum_assignment(cost)

    prev_map = {name: set(nodes) for name, nodes in prev_mods}
    curr_map = {name: set(nodes) for name, nodes in cur_mods}
    prev_names = [m[0] for m in prev_mods]
    curr_names = [m[0] for m in cur_mods]

    turnovers = []
    matches = []
    for pi, cj in zip(ri, ci):
        pn, cn = prev_names[pi], curr_names[cj]
        a, b = prev_map[pn], curr_map[cn]
        inter = len(a & b)
        uni = len(a | b) if (a or b) else 1
        jacc_turnover = 1.0 - inter / uni
        turnovers.append(jacc_turnover)
        matches.append((pn, cn, float(S_sim[pi, cj])))

    mean_turnover = float(np.mean(turnovers)) if turnovers else float("nan")
    return mean_turnover, matches

# ====== NULLS & BOOTSTRAPS ======
def degree_preserving_rewire(S, n_swaps=5_000, rng=None):
    """
    Maslov-Sneppen style on unweighted graph defined by |S|>0, then
    restore sign/weights by aligning with original magnitudes (rank-based).
    For speed/memory: operate on edge list; nodes up to ~1k are fine.
    """
    rng = np.random.default_rng(None if rng is None else rng)
    A = (np.abs(S) > 0).astype(np.uint8)
    np.fill_diagonal(A, 0)
    # edge list (i<j)
    ii, jj = np.where(np.triu(A, 1))
    edges = np.stack([ii, jj], axis=1)
    if edges.shape[0] < 2:
        return S.copy()

    # perform random double-edge swaps
    for _ in range(min(n_swaps, edges.shape[0]*10)):
        a, b = rng.integers(0, edges.shape[0], size=2)
        if a == b: continue
        i, j = edges[a]
        u, v = edges[b]
        if i==u or i==v or j==u or j==v: continue
        # propose (i,v) and (u,j)
        if i==v or u==j: continue
        # avoid duplicates
        if A[i, v] or A[u, j]: continue
        # do swap
        A[i, j] = A[j, i] = 0
        A[u, v] = A[v, u] = 0
        A[i, v] = A[v, i] = 1
        A[u, j] = A[j, u] = 1
        edges[a] = [i, v]
        edges[b] = [u, j]

    # rebuild S': keep original absolute weights sorted and assign to new edges
    absw = np.abs(np.triu(S, 1))
    wvals = absw[absw > 0].ravel()
    wvals.sort()
    ii2, jj2 = np.where(np.triu(A, 1))
    m = len(ii2)
    if m == 0: return np.zeros_like(S)
    # match largest weights to current edges
    assign = np.zeros_like(S, dtype=S.dtype)
    take = min(m, wvals.size)
    # fill from largest
    for k in range(take):
        i, j = int(ii2[k]), int(jj2[k])
        w = wvals[-(k+1)]
        assign[i, j] = assign[j, i] = w
    # randomize signs according to original sign proportion
    pos_frac = float((S > 0).sum()) / (S.size + 1e-12)
    signs = np.where(np.random.rand(*assign.shape) < pos_frac, 1.0, -1.0)
    Sprime = assign * signs
    np.fill_diagonal(Sprime, 0.0)
    return Sprime

def bootstrap_ci(x, alpha=0.05, B=500, rng=None):
    rng = np.random.default_rng(None if rng is None else rng)
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    bs = []
    n = x.size
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        bs.append(float(np.mean(x[idx])))
    lo = float(np.percentile(bs, 100*alpha/2))
    hi = float(np.percentile(bs, 100*(1-alpha/2)))
    return float(np.mean(x)), (lo, hi)

def continuity_nulls(prev_mods, prev_S, cur_S, B=100, rng=None):
    """
    Rewire-based null for Jaccard turnover under 1-1 Hungarian mapping.
    Returns list of turnovers across B rewired samples.
    """
    vals = []
    for b in range(B):
        Snull = degree_preserving_rewire(cur_S, n_swaps=2_000, rng=(None if rng is None else rng+b))
        # build "current" modules on rewired S using greedy modularity
        Gnull = nx.Graph()
        d = Snull.shape[0]
        Gnull.add_nodes_from(range(d))
        ii, jj = np.where(np.triu(np.abs(Snull) > 0, 1))
        for i, j in zip(ii, jj):
            Gnull.add_edge(int(i), int(j), weight=float(Snull[i, j]))
        cur_mods_null = greedy_modules(Gnull)
        tnull, _ = continuity_hungarian(prev_mods, cur_mods_null, prev_S, Snull)
        if np.isfinite(tnull):
            vals.append(float(tnull))
    return vals
def save_top_module_attention_map(model, ref_cache, mods, layer_H, out_png):
    """
    Picks the top module by attention-weighted score (using the first ref batch),
    then produces a contact sheet PNG highlighting high-score patches.
    """
    if not mods: 
        return
    Xr = ref_cache[0][0].to(next(model.parameters()).device)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        _ = model(Xr)
    # rank modules
    ranks = attention_weighted_module_scores(model, Xr, mods, np.asarray(layer_H, dtype=np.float32))
    if not ranks: 
        return
    top_mod = ranks[0]["module"]
    # compute per-patch score for that module only
    name2nodes = {name: nodes for name, nodes in mods}
    nodes = name2nodes.get(top_mod, [])
    if not nodes:
        return
    # activation-based per-patch score for this module
    H = np.asarray(layer_H, dtype=np.float32)
    mscore = np.abs(H[:, nodes]).mean(axis=1)  # [num_patches]
    mscore = (mscore - mscore.min()) / (mscore.ptp() + 1e-12)
    # reuse contact sheet painter
    scores_t = torch.from_numpy(mscore.astype(np.float32))
    contact_sheet_with_heat(Xr.detach().cpu(), scores_t, out_png, order_by="score", cols=8)

# --- strength-weighted centroids (was uniform) ---
def module_centroids(S, modules):
    d = S.shape[0]
    C = np.zeros((len(modules), d), dtype=float)
    Wabs = np.abs((S + S.T) / 2.0)
    strength = Wabs.sum(axis=1).astype(float)  # node strength
    for i, (_, nodes) in enumerate(modules):
        if not nodes: 
            continue
        w = strength[nodes]
        if w.sum() <= 0:
            C[i, nodes] = 1.0 / math.sqrt(len(nodes))
        else:
            C[i, nodes] = w / (np.linalg.norm(w) + 1e-12)
    return C

def cosine_sim_matrix(U, V):
    U = U + 1e-12; V = V + 1e-12
    Un = U / np.linalg.norm(U, axis=1, keepdims=True)
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    return Un @ Vn.T

def match_modules(prev_mods, cur_mods, prev_S, cur_S):
    if not prev_mods or not cur_mods: return {}
    U = module_centroids(prev_S, prev_mods)
    V = module_centroids(cur_S, cur_mods)
    S = cosine_sim_matrix(U, V)
    cost = 1.0 - S
    r, c = linear_sum_assignment(cost)
    mapping = {}
    for i, j in zip(r, c):
        mapping[prev_mods[i][0]] = (cur_mods[j][0], float(S[i, j]))
    return mapping

def plot_graph_image(G, out_png, title="", seed=1337):
    if G.number_of_edges() == 0:
        fig = plt.figure(figsize=(4,4)); plt.axis("off"); plt.title(title); fig.savefig(out_png, dpi=200); plt.close(); return
    H = G.copy()
    pos = nx.spring_layout(H, seed=seed, dim=2, iterations=50)
    ws = np.array([abs(H[u][v].get("weight", 0.0)) for u, v in H.edges()], dtype=float)
    wmin, wrng = (float(ws.min()), float(np.ptp(ws))) if ws.size else (0.0, 0.0)
    widths = 0.5 + 2.5 * ((ws - wmin) / (wrng + 1e-8)) if ws.size else np.array([1.0])
    plt.figure(figsize=(6,6))
    nx.draw_networkx_nodes(H, pos, node_size=8, alpha=0.8)
    nx.draw_networkx_edges(H, pos, width=widths, alpha=0.25)
    plt.axis("off"); 
    if title: plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()

def plot_epoch_dashboard(layer_pngs, out_png, epoch, metrics_dict):
    from PIL import Image, ImageDraw
    w = 900; h = 300
    canv = Image.new("RGB", (w*len(layer_pngs), h), (255,255,255))
    for i, p in enumerate(layer_pngs):
        if not os.path.exists(p): continue
        im = Image.open(p).convert("RGB").resize((w, h))
        canv.paste(im, (i*w, 0))
    txt = f"epoch={epoch}  " + "  ".join([f"{k}={v:.4f}" for k,v in metrics_dict.items() if isinstance(v,(int,float)) and np.isfinite(v)])
    draw = ImageDraw.Draw(canv); draw.text((10, 10), txt, fill=(0,0,0))
    canv.save(out_png)

def cka_linear(X, Y):
    if X is None or Y is None: return np.nan
    X = np.asarray(X, dtype=float); Y = np.asarray(Y, dtype=float)
    X -= X.mean(0, keepdims=True); Y -= Y.mean(0, keepdims=True)
    Kx = X @ X.T; Ky = Y @ Y.T
    def hsic(K, L):
        Kc = K - K.mean(0, keepdims=True) - K.mean(1, keepdims=True) + K.mean()
        Lc = L - L.mean(0, keepdims=True) - L.mean(1, keepdims=True) + L.mean()
        return (Kc*Lc).sum()
    num = hsic(Kx, Ky)
    den = math.sqrt(hsic(Kx, Kx) * hsic(Ky, Ky) + 1e-12)
    return float(num / (den + 1e-12))

# --- weighted degree entropy (was unweighted) ---
def degree_entropy(G):
    if G.number_of_nodes()==0: return 0.0
    degs = np.array([d for _, d in G.degree(weight="weight")], dtype=float)
    if degs.sum() <= 0: return 0.0
    p = degs / degs.sum()
    return float(-(p * np.log(p + 1e-12)).sum())

def pca_entropy_and_pr(A, max_r=128):
    X = np.asarray(A, dtype=float)
    X = zscore_cols(X)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    lam = (S**2)
    if max_r is not None and max_r < lam.shape[0]:
        lam = lam[:max_r]
    lam = np.clip(lam, 1e-12, None)
    p = lam / lam.sum()
    H = float(-(p * np.log(p)).sum())
    PR = float((lam.sum()**2) / (np.square(lam).sum() + 1e-12))
    return H, PR

def ckpt_sha1(model):
    h = hashlib.sha1()
    with torch.no_grad():
        for k, v in model.state_dict().items():
            h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

# -----------------------------
# Spectral metrics (unchanged)
# -----------------------------
def laplacian_spectrum_metrics(S, modules):
    W = np.abs((S + S.T) / 2.0)
    np.fill_diagonal(W, 0.0)
    d = W.sum(axis=1)
    L = np.diag(d) - W
    with np.errstate(divide='ignore'):
        d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    Dinv = np.diag(d_inv_sqrt)
    Lnorm = Dinv @ L @ Dinv
    try:
        lam = np.linalg.eigvalsh((Lnorm + Lnorm.T)/2.0)
    except np.linalg.LinAlgError:
        lam = np.linalg.eigvals((Lnorm + Lnorm.T)/2.0).real
    lam = np.clip(lam, 1e-12, None)
    p = lam / lam.sum()
    Hspec = float(-(p * np.log(p)).sum())
    k0 = int(np.sum(lam < 1e-6))
    global_dict = {"lap_spec_entropy": Hspec, "lap_connected_components": k0,
                   "lap_lambda_min": float(lam.min()), "lap_lambda_max": float(lam.max()),
                   "lap_lambda_mean": float(lam.mean())}
    mod_rows = []
    for name, nodes in modules:
        idx = np.ix_(nodes, nodes)
        Wm = W[idx]
        dm = Wm.sum(axis=1)
        Lm = np.diag(dm) - Wm
        with np.errstate(divide='ignore'):
            dinv = 1.0 / np.sqrt(np.maximum(dm, 1e-12))
        Dm = np.diag(dinv)
        LmN = Dm @ Lm @ Dm
        if LmN.size == 0:
            continue
        try:
            lam_m = np.linalg.eigvalsh((LmN + LmN.T)/2.0)
        except np.linalg.LinAlgError:
            lam_m = np.linalg.eigvals((LmN + LmN.T)/2.0).real
        lam_m = np.clip(lam_m, 1e-12, None)
        pm = lam_m / lam_m.sum()
        Hm = float(-(pm * np.log(pm)).sum())
        k0m = int(np.sum(lam_m < 1e-6))
        mod_rows.append({"module": name, "size": len(nodes), "lap_spec_entropy": Hm,
                         "lap_connected_components": k0m,
                         "lap_lambda_min": float(lam_m.min()),
                         "lap_lambda_max": float(lam_m.max()),
                         "lap_lambda_mean": float(lam_m.mean())})
    return global_dict, mod_rows

# -----------------------------
# 3D spectral coordinates (stable across epochs)
# -----------------------------
def spectral_coords_3d(S: np.ndarray, prev: dict, L: int) -> np.ndarray:
    W = np.abs((S + S.T) / 2.0)
    np.fill_diagonal(W, 0.0)
    d = W.sum(axis=1)
    Lm = np.diag(d) - W
    with np.errstate(divide='ignore'):
        dinv = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    Dm = np.diag(dinv)
    Lnorm = Dm @ Lm @ Dm
    lam, U = np.linalg.eigh((Lnorm + Lnorm.T) / 2.0)
    idx = np.argsort(lam)
    U = U[:, idx]
    if U.shape[1] >= 4:
        coords = U[:, 1:4]
    else:
        k = min(3, U.shape[1])
        coords = U[:, :k]
        if k < 3:
            z = np.zeros((coords.shape[0], 3 - k), dtype=coords.dtype)
            coords = np.concatenate([coords, z], axis=1)

    if "specU" not in prev:
        prev["specU"] = {}
    if L not in prev["specU"] or prev["specU"][L] is None:
        prev["specU"][L] = coords.copy()
        return coords

    # Procrustes-align
    A = coords
    B = prev["specU"][L]
    A0 = A - A.mean(axis=0, keepdims=True)
    B0 = B - B.mean(axis=0, keepdims=True)
    M = A0.T @ B0
    Uo, _, Vo = np.linalg.svd(M, full_matrices=False)
    R = Uo @ Vo
    A_aligned = A0 @ R
    return A_aligned

def plot_graph_3d(G: nx.Graph, coords: np.ndarray, out_png: str, title: str = ""):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    X, Y, Z = coords[:, 0], coords[:, 1], coords[:, 2]
    ax.scatter(X, Y, Z, s=6, alpha=0.9)
    for u, v, data in G.edges(data=True):
        w = abs(float(data.get("weight", 0.0)))
        xs = [X[u], X[v]]; ys = [Y[u], Y[v]]; zs = [Z[u], Z[v]]
        ax.plot(xs, ys, zs, linewidth=0.5 + 2.0 * w, alpha=0.2)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

# -----------------------------
# Grad-CAM style patch attribution
# -----------------------------
def gradcam_patch_attribution(model: nn.Module, X: torch.Tensor):
    model.eval()
    with torch.no_grad():
        H = model.encode_patches(X)  # [S, D]
    H = H.detach().requires_grad_(True)
    z, w = model.mil(H)
    y = model.head(z).view(-1)
    target = y.sum()
    model.zero_grad(set_to_none=True)
    target.backward()
    g = H.grad.detach()
    scores = (H * g).sum(dim=1)
    scores = torch.relu(scores)
    s = scores - scores.min()
    if float(s.max()) > 1e-12:
        s = s / s.max()
    return y.detach(), w.detach().squeeze(-1), s.detach()

def partition_with_singletons(G: nx.Graph, modules):
    comms = [set(nodes) for _, nodes in modules if len(nodes) > 0]
    seen = set()
    disjoint = []
    for c in comms:
        c = c - seen
        if c:
            disjoint.append(c)
            seen |= c
    all_nodes = set(G.nodes())
    missing = all_nodes - seen
    disjoint.extend([{u} for u in sorted(missing)])
    return disjoint

def contact_sheet_with_heat(X: torch.Tensor, scores: torch.Tensor, out_png: str, order_by: str = "score", k: int | None = None, cols: int = 8):
    from PIL import Image, ImageDraw
    def _denorm(img_t: torch.Tensor) -> np.ndarray:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(img_t.device)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(img_t.device)
        x = (img_t * std + mean).clamp(0,1).cpu().permute(1,2,0).numpy()
        return (x * 255).astype(np.uint8)
    N = X.size(0)
    idx = torch.arange(N)
    if order_by == "score":
        idx = torch.argsort(scores, descending=True)
    if k is not None:
        idx = idx[:k]
    idx = idx.cpu().tolist()
    rows = (len(idx) + cols - 1) // cols
    cell = 224
    from PIL import Image
    canv = Image.new("RGB", (cols*cell, rows*cell), (255,255,255))
    for i, j in enumerate(idx):
        r, c = divmod(i, cols)
        img = Image.fromarray(_denorm(X[j]))
        draw = ImageDraw.Draw(img)
        s = float(scores[j].cpu()); w = max(1, int(6*s + 1))
        color = (int(255*s), 0, int(255*(1-s)))
        for t in range(w):
            draw.rectangle([t, t, img.size[0]-1-t, img.size[1]-1-t], outline=color)
        canv.paste(img, (c*cell, r*cell))
    canv.save(out_png)

@torch.no_grad()
def delta_pred_and_auc(model: nn.Module, X: torch.Tensor, y_true_scalar: float | None, scores: torch.Tensor, top_frac: float = 0.10, auc_threshold: float | None = None):
    model.eval()
    with autocast():
        y0, _, _ = model(X)
    y0 = float(y0.view(-1).cpu().numpy())
    N = scores.numel(); k = max(1, int(round(top_frac * N)))
    order = torch.argsort(scores, descending=True)
    mask_idx = order[:k].cpu().tolist()
    Xm = X.clone()
    Xm[mask_idx] = 0.0
    with autocast():
        y1, _, _ = model(Xm)
    y1 = float(y1.view(-1).cpu().numpy())
    delta = y1 - y0
    delta_auc = None
    return y0, y1, delta, delta_auc

# -----------------------------
# Attention-weighted module scores
# -----------------------------
def attention_weighted_module_scores(model: nn.Module, X: torch.Tensor, modules, layer_H: np.ndarray) -> list[dict]:
    model.eval()
    with torch.no_grad(), autocast(enabled=True):
        H_final = model.encode_patches(X)  # [S,D]
        _, w = model.mil(H_final)
    w = w.detach().float().cpu().view(-1).numpy()
    H = np.asarray(layer_H, dtype=float)
    if H.shape[0] != w.shape[0]:
        m = min(H.shape[0], w.shape[0])
        H = H[:m]; w = w[:m]
    rows = []
    for name, nodes in modules:
        if len(nodes) == 0: continue
        Hm = np.abs(H[:, nodes]).mean(axis=1)
        score = float(np.mean(w * Hm))
        rows.append({"module": name, "attn_weighted_score": score, "size": len(nodes)})
    rows.sort(key=lambda r: r["attn_weighted_score"], reverse=True)
    return rows

# -----------------------------
# Soft memberships + OT transport
# -----------------------------
def soft_kmeans(X: np.ndarray, K: int, iters: int = 30, temp: float = 0.5, seed: int = 0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    if K > n: K = n
    C = X[rng.choice(n, size=K, replace=False)]
    for _ in range(iters):
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        S = Xn @ Cn.T  # cosine similarity
        R = np.exp(S / max(1e-6, temp))
        R = R / (R.sum(axis=1, keepdims=True) + 1e-12)
        C = (R.T @ X) / (R.sum(axis=0, keepdims=True).T + 1e-12)
    M = R  # rows sum to 1
    return M, C

def spectral_membership(S: np.ndarray, K: int, seed: int = 0):
    W = np.abs((S + S.T)/2.0); np.fill_diagonal(W, 0.0)
    d = W.sum(axis=1); L = np.diag(d) - W
    dinv = 1.0 / np.sqrt(np.maximum(d, 1e-12)); D = np.diag(dinv)
    Ln = D @ L @ D
    lam, U = np.linalg.eigh((Ln + Ln.T)/2.0)
    idx = np.argsort(lam)
    r = min(16, max(3, int(np.ceil(np.log2(U.shape[0]+1)))))
    X = U[:, idx[:r]]
    M, C = soft_kmeans(X, K=K, iters=30, temp=0.5, seed=seed)
    return M, C, X

def sinkhorn_transport(a, b, C, eps=0.05, iters=200):
    # a,b >=0, sum to 1
    K = np.exp(-C/eps)
    u = np.ones_like(a); v = np.ones_like(b)
    for _ in range(iters):
        u = a / (K @ v + 1e-12)
        v = b / (K.T @ u + 1e-12)
    T = np.diag(u) @ K @ np.diag(v)
    return T

def module_transport(M_prev: np.ndarray, M_cur: np.ndarray):
    a = M_prev.sum(axis=0); b = M_cur.sum(axis=0)
    a = a / (a.sum() + 1e-12); b = b / (b.sum() + 1e-12)
    def centroids_from_M(M):
        return M.T / (M.sum(axis=0, keepdims=True).T + 1e-12)
    C_prev = centroids_from_M(M_prev)  # [Kp, d]
    C_cur  = centroids_from_M(M_cur)   # [Kc, d]
    Un = C_prev / (np.linalg.norm(C_prev, axis=1, keepdims=True) + 1e-12)
    Vn = C_cur  / (np.linalg.norm(C_cur,  axis=1, keepdims=True) + 1e-12)
    S = Un @ Vn.T
    C = 1.0 - np.clip(S, -1.0, 1.0)
    T = sinkhorn_transport(a, b, C, eps=0.05, iters=200)
    return T, C

# -----------------------------
# Activation logger (fp16 buffers to save RAM)
# -----------------------------
class ActivationLogger:
    """
    Robust activation logger:
      - If model.backbone.blocks exists (ModuleList), hook those indices.
      - Else, auto-discover leaf layers in model.backbone (Conv/Linear/MHAttn) and
        allow selecting by index (depth order) or by (sub)string name.
      - Stores CLS token when a [B,T,D] tensor is seen; else flattens [B,*] → [B,D].
      - Buffers are kept on CPU in float16 to reduce RAM.
    """
    def __init__(self, model: nn.Module, layers):
        self.hooks = []
        self.buffers = {}
        self.targets = []  # list[(name, module)]
        bb = getattr(model, "backbone", model)  # fall back to whole model

        if hasattr(bb, "blocks") and isinstance(getattr(bb, "blocks"), (nn.ModuleList, list)):
            self.mode = "blocks"
            blocks = list(bb.blocks)
            self.targets = [(f"backbone.blocks.{i}", m) for i, m in enumerate(blocks)]
        else:
            self.mode = "leaves"
            leaves = []
            for name, mod in bb.named_modules():
                if not name:
                    continue
                if any(True for _ in mod.children()):
                    continue
                if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear, nn.MultiheadAttention)):
                    leaves.append((f"backbone.{name}", mod))
            if not leaves:
                leaves = [(f"backbone.{n}", m) for n, m in bb.named_children()]
            self.targets = leaves

        self.sel = self._resolve_selection(layers)

        for idx in self.sel:
            name, module = self.targets[idx]
            self.buffers[idx] = []
            self.hooks.append(module.register_forward_hook(self._make_hook(idx, name)))

        chosen = [self.targets[i][0] for i in self.sel]
        print(f"[ActivationLogger] hooked {len(chosen)} layer(s): {chosen}", flush=True)

    def _resolve_selection(self, layers):
        if isinstance(layers, str):
            items = [x.strip() for x in layers.split(",") if x.strip() != ""]
        elif isinstance(layers, (list, tuple)):
            items = list(layers)
        else:
            items = [layers]
        all_int = False
        try:
            _ = [int(x) for x in items]
            all_int = True
        except Exception:
            all_int = False
        if all_int:
            idxs = [int(x) for x in items]
            idxs = [i for i in idxs if 0 <= i < len(self.targets)]
            if not idxs:
                mid = min(len(self.targets) // 2, max(0, len(self.targets) - 1))
                idxs = [mid]
            return idxs
        matched = set()
        lower_names = [n.lower() for n, _ in self.targets]
        for token in items:
            t = str(token).lower()
            for i, nm in enumerate(lower_names):
                if t in nm:
                    matched.add(i)
        if not matched:
            mid = min(len(self.targets) // 2, max(0, len(self.targets) - 1))
            matched = {mid}
        return sorted(matched)

    def _first_tensor(self, out):
        if torch.is_tensor(out):
            return out
        if isinstance(out, (list, tuple)):
            for x in out:
                if torch.is_tensor(x):
                    return x
        return None

    def _make_hook(self, idx, name):
        def hook(module, inp, out):
            t = self._first_tensor(out)
            if t is None:
                return
            h = t
            if h.dim() >= 3 and h.size(1) > 1:
                h = h[:, 0, :]
            elif h.dim() > 2:
                h = h.reshape(h.size(0), -1)
            self.buffers[idx].append(h.detach().to("cpu", dtype=torch.float16))
        return hook

    def clear_epoch(self):
        for k in self.buffers:
            self.buffers[k].clear()

    def get_epoch_activations(self):
        acts = {}
        for k, chunks in self.buffers.items():
            if not chunks:
                continue
            H = torch.cat(chunks, dim=0)  # [N, D] fp16
            acts[k] = H.numpy()           # keep fp16 as numpy to save RAM
        return acts

    def remove(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks.clear()


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches_dir", required=True, type=str)
    ap.add_argument("--labels_csv", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_patches", type=int, default=64)   # lean default
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-2)
    ap.add_argument("--freeze_blocks", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layers", type=str, default="1,3,5")
    ap.add_argument("--graph_topk", type=int, default=15)
    ap.add_argument("--ref_batches", type=int, default=2)    # lean default
    ap.add_argument("--attr_top_frac", type=float, default=0.10)
    ap.add_argument("--save_spatial", action="store_true")
    ap.add_argument("--spatial_examples", type=int, default=4)
    ap.add_argument("--auc_threshold", type=float, default=None)
    # Hard caps for RAM before graph/SVD
    ap.add_argument("--cap_patches",  type=int, default=2000, help="Max rows from H per layer.")
    ap.add_argument("--cap_features", type=int, default=512,  help="Random-projected cols per layer.")
    args = ap.parse_args()

    # ===== ABLATIONS (flip here to run a quick ablation pass; no new CLI flags) =====
    ABLATION_FREEZE_ROWS      = True   # False → re-sample rows each epoch
    ABLATION_SOFT_MEMBERSHIPS = True   # False → use only greedy communities (hard)
    ABLATION_TEMPORAL_REG     = True   # False → no stickiness blend for M_t
    ABLATION_USE_OT           = True   # False → disable OT lineage; keep Hungarian baseline
    N_NULL_BOOTSTRAPS         = 50     # 0 → skip null rewires/CI per epoch (cheaper)

    seed_everything(args.seed)
    ensure_dir(args.out_dir)
    log_path = Path(args.out_dir) / "_run.log"
    with open(log_path, "w") as f: f.write("run_start\n")

    # -------- LABELS --------
    def normalize_case_id(labels: pd.DataFrame) -> pd.DataFrame:
        """
        Robustly create a single lowercase 'case_id' column even if the CSV has
        duplicated headers or weird spacing. Picks the FIRST positional match.
        """
        if not isinstance(labels, pd.DataFrame):
            raise TypeError("labels must be a DataFrame")

    # normalize header names
        cols_norm = [str(c).strip().lower().replace(" ", "_") for c in labels.columns]
        labels = labels.copy()
        labels.columns = cols_norm

    # candidates in priority order
        candidates = ("case_id", "case", "caseid", "patient", "tcga_id")

        chosen = None
        for c in candidates:
            if c in labels.columns:
                chosen = c
                break
        if chosen is None:
            # fallback: any column containing these tokens
            toks = ("case", "patient", "tcga")
            for i, c in enumerate(labels.columns):
                if any(t in c for t in toks):
                    chosen = c
                    break
        if chosen is None:
            raise ValueError(f"No case-id column found. Columns={list(labels.columns)}")

    # if header duplicated, labels[chosen] would be a DataFrame → use first match by position
        first_pos = next(i for i, c in enumerate(labels.columns) if c == chosen)
        s = labels.iloc[:, first_pos].astype(str).map(lambda x: x.strip().lower())
        s.name = "case_id"

    # drop duplicate header columns so we have a single 'case_id'
        labels = labels.loc[:, ~labels.columns.duplicated()].copy()
        labels["case_id"] = s

    # sanity checks
        if labels["case_id"].isna().all():
            raise ValueError("case_id column is empty after normalization")
        return labels


    labels = pd.read_csv(args.labels_csv)
    labels = normalize_case_id(labels)
    if "rt_mean" not in labels.columns:
        raise ValueError("labels need 'rt_mean'")
    targets = labels.groupby("case_id")[["rt_mean"]].mean().reset_index()
    case_targets = {r["case_id"]: r[["rt_mean"]].to_numpy(dtype=np.float32) for _, r in targets.iterrows()}

    # -------- SLIDES --------
    patches_dir = Path(args.patches_dir)
    if not patches_dir.exists():
        raise FileNotFoundError(f"patches_dir does not exist: {patches_dir}")

    slide_dirs = sorted([d for d in os.listdir(patches_dir) if (patches_dir / d).is_dir()])
    slide_case_ids = [d[:12].lower() for d in slide_dirs]
    keep = [cid in case_targets for cid in slide_case_ids]
    slide_dirs = [s for s, k in zip(slide_dirs, keep) if k]
    slide_case_ids = [c for c, k in zip(slide_case_ids, keep) if k]
    counts = [len(list((patches_dir / s).glob("*.png"))) for s in slide_dirs]
    has_png = [c > 0 for c in counts]
    slide_dirs = [s for s, h in zip(slide_dirs, has_png) if h]
    slide_case_ids = [c for c, h in zip(slide_case_ids, has_png) if h]

    with open(log_path, "a") as f:
        f.write(f"labels_csv={args.labels_csv}\n")
        f.write(f"patches_dir={args.patches_dir}\n")
        f.write(f"out_dir={args.out_dir}\n")
        f.write(f"n_targets={len(case_targets)}\n")
        f.write(f"n_slide_dirs_total={len(os.listdir(patches_dir))}\n")
        f.write(f"n_slide_dirs_after_case_filter={len(slide_dirs)}\n")
        f.write(f"n_slides_with_png={sum(has_png) if has_png else 0}\n")

    if len(slide_dirs) == 0:
        raise RuntimeError("No slides left after filtering. Check case IDs and that PNGs exist.")

    # -------- SPLIT --------
    n = len(slide_dirs)
    idx = np.arange(n)
    rng = np.random.default_rng(args.seed); rng.shuffle(idx)
    cut = int(0.8 * n)
    tr_idx, va_idx = idx[:cut], idx[cut:]
    tr_slides = [slide_dirs[i] for i in tr_idx]
    va_slides = [slide_dirs[i] for i in va_idx]
    tr_cases  = [slide_case_ids[i] for i in tr_idx]
    va_cases  = [slide_case_ids[i] for i in va_idx]
    if len(tr_slides) == 0 or len(va_slides) == 0:
        raise RuntimeError(f"Train/val split empty. n={n}, tr={len(tr_slides)}, va={len(va_slides)}")

    # -------- DATASETS / LOADERS (RAM conservative) --------
    train_ds = SlideBagDataset(tr_slides, tr_cases, case_targets, str(patches_dir), max_patches=args.max_patches, train=True)
    valid_ds = SlideBagDataset(va_slides, va_cases, case_targets, str(patches_dir), max_patches=args.max_patches, train=False)
    dl_kwargs = dict(num_workers=0, pin_memory=False, persistent_workers=False, collate_fn=safe_collate)
    train_loader = DataLoader(train_ds, batch_size=max(1, args.batch_size), shuffle=True,  **dl_kwargs)
    valid_loader = DataLoader(valid_ds, batch_size=1,                            shuffle=False, **dl_kwargs)
    if len(train_loader) == 0: raise RuntimeError("Empty train_loader.")
    if len(valid_loader) == 0: raise RuntimeError("Empty valid_loader.")

    # -------- MODEL / OPTIM --------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WSItoRT(out_dim=1, freeze_backbone_blocks=args.freeze_blocks).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    scaler  = GradScaler()

    # -------- REFERENCE CACHE (fixed across epochs) --------
    def ref_iter(loader, limit):
        count = 0
        for b in loader:
            if b is None or (isinstance(b, (list, tuple)) and b[0] is None): continue
            yield b
            count += 1
            if count >= limit: break
    ref_cache = []
    for (X, y, cid, sdir) in ref_iter(valid_loader, args.ref_batches):
        ref_cache.append((X.cpu(), y.cpu() if y is not None else None, cid, sdir))
    if not ref_cache:
        raise RuntimeError("No reference batches available from valid_loader.")

    # -------- HOOKS --------
    actlog = ActivationLogger(model, layers=args.layers)
    selected_idxs = actlog.sel
    print(f"[ActivationLogger] selected indices: {selected_idxs}", flush=True)

    # -------- PREV/STATE CACHES --------
    prev = {
        "modules": {L: [] for L in selected_idxs},
        "S":       {L: None for L in selected_idxs},
        "A":       {L: None for L in selected_idxs},
        "row_idx": {},      # fixed row subset per layer
        "RP":      {},      # per-layer RP matrix
        "specU":   {},      # spectral basis for stable 3D coords
        "M":       {L: None for L in selected_idxs},   # soft memberships
        "Pi":      {L: None for L in selected_idxs},   # transport for stickiness
        "hist_mods": {L: [] for L in selected_idxs},   # for optional epoch-shuffle null
    }
    metrics_rows = []

    # =========================
    # TRAIN / EVAL LOOP
    # =========================
    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        tot, ntr = 0.0, 0
        for b in train_loader:
            if b is None or (isinstance(b, (list, tuple)) and b[0] is None): continue
            X, y, _, _ = b
            X = X.to(device); y = y.to(device).view(-1)
            opt.zero_grad(set_to_none=True)
            with autocast():
                yhat, _, _ = model(X)
                loss = loss_fn(yhat, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item(); ntr += 1
        tr_loss = tot / max(1, ntr)

        # ---- Quick Val ----
        model.eval()
        with torch.no_grad():
            nva, vtot = 0, 0.0
            for i, b in enumerate(valid_loader):
                if b is None or (isinstance(b, (list, tuple)) and b[0] is None): continue
                X, y, _, _ = b
                X = X.to(device); y = y.to(device).view(-1)
                with autocast():
                    yhat, _, _ = model(X)
                    l = loss_fn(yhat, y)
                vtot += l.item(); nva += 1
                if i >= 16: break
            va_loss = vtot / max(1, nva)

        # ---- Activate hooks on fixed reference ----
        actlog.clear_epoch()
        for (Xc, _yc, _cid, _sdir) in ref_cache:
            with torch.no_grad(), autocast(enabled=True):
                _ = model(Xc.to(device))
        acts = actlog.get_epoch_activations()

        # ---- Epoch Dir ----
        ep_dir = Path(args.out_dir) / f"epoch_{epoch:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        base_metrics = {"epoch": epoch, "train_loss": float(tr_loss), "val_loss": float(va_loss)}
        (ep_dir / "metrics.json").write_text(json.dumps(base_metrics, indent=2))
        with open(log_path, "a") as f: f.write(f"epoch={epoch} ep_dir={ep_dir}\n")

        layer_pngs = []
        for L in selected_idxs:
            H = acts.get(L, None)
            if H is None or H.size == 0:
                continue
            H = H.astype(np.float32, copy=False)

            # ---- Caps with optional row-freeze ablation ----
            if H.shape[0] > args.cap_patches:
                if ABLATION_FREEZE_ROWS:
                    idx_rows = prev["row_idx"].get(L)
                    if idx_rows is None or idx_rows.size == 0 or int(np.max(idx_rows)) >= H.shape[0]:
                        rng_rows = np.random.default_rng(args.seed + 12345 + L)
                        take = min(args.cap_patches, H.shape[0])
                        idx_rows = np.sort(rng_rows.choice(H.shape[0], size=take, replace=False))
                        prev["row_idx"][L] = idx_rows
                    else:
                        idx_rows = idx_rows[idx_rows < H.shape[0]]
                        if idx_rows.size == 0:
                            idx_rows = np.arange(min(args.cap_patches, H.shape[0]))
                            prev["row_idx"][L] = idx_rows
                else:
                    rng_rows = np.random.default_rng(args.seed + 12345 + L + epoch)
                    take = min(args.cap_patches, H.shape[0])
                    idx_rows = np.sort(rng_rows.choice(H.shape[0], size=take, replace=False))
                H = H[idx_rows]

            if H.shape[1] > args.cap_features:
                if L not in prev["RP"]:
                    rng_cols = np.random.default_rng(args.seed + 9999 + L)
                    P = rng_cols.standard_normal(size=(H.shape[1], args.cap_features)).astype(np.float32)
                    P /= np.sqrt(H.shape[1])
                    prev["RP"][L] = P
                H = H @ prev["RP"][L]

            # ---- Graph & modules ----
            G, S = build_graph_from_activations(H, topk=args.graph_topk)
            mods = greedy_modules(G)

            # 2D + 3D plots
            png2d = str(ep_dir / f"layer_L{L}.png")
            plot_graph_image(G, png2d, title=f"L{L} |E|={G.number_of_edges()}")
            layer_pngs.append(png2d)

            coords3d = spectral_coords_3d(S, prev, L)
            png3d = str(ep_dir / f"layer_L{L}_3d.png")
            plot_graph_3d(G, coords3d, png3d, title=f"L{L} 3D |E|={G.number_of_edges()}")

            # Save matrices (compressed)
            np.savez_compressed(ep_dir / f"A_L{L}.npz", A=H)
            np.savez_compressed(ep_dir / f"W_L{L}.npz", W=S)

            # ---- Soft memberships (+ stickiness via OT) or hard-only ablation ----
            M_t = None
            if ABLATION_SOFT_MEMBERSHIPS:
                d_nodes = S.shape[0]
                K = min(8, max(2, int(np.ceil(np.sqrt(max(1, d_nodes)//20 + 1)))))
                M_t, _C_t, _Xeig = spectral_membership(S, K, seed=args.seed + L)
                if ABLATION_TEMPORAL_REG and prev["M"][L] is not None and prev["M"][L].shape[1] > 0:
                    if ABLATION_USE_OT:
                        Tmat, _C = module_transport(prev["M"][L], M_t)
                        Pi = Tmat / (Tmat.sum(axis=0, keepdims=True) + 1e-12)
                        alpha = 0.2
                        M_t = (1 - alpha) * M_t + alpha * (prev["M"][L] @ Pi)
                        M_t = M_t / (M_t.sum(axis=1, keepdims=True) + 1e-12)
                        prev["Pi"][L] = Pi
                        # lineage edges
                        thr = 0.2
                        edges = [(i, j, float(Tmat[i, j])) for i in range(Tmat.shape[0]) for j in range(Tmat.shape[1]) if Tmat[i, j] >= thr]
                        if edges:
                            pd.DataFrame(edges, columns=["prev_k","curr_k","flow"]).to_csv(ep_dir / f"lineage_L{L}.csv", index=False)
                if M_t is not None:
                    np.savez_compressed(ep_dir / f"M_L{L}.npz", M=M_t.astype(np.float32))

            # ---- Legacy Hungarian baseline (1-1) ----
            mapping = {}
            if prev["S"][L] is not None and prev["modules"][L]:
                mapping = match_modules(prev["modules"][L], mods, prev["S"][L], S)
                pd.DataFrame([{"prev": k, "curr": v[0], "cos": v[1]} for k, v in mapping.items()]).to_csv(ep_dir / f"match_L{L}.csv", index=False)

            # ---- Module tables + spectral metrics ----
            rows = [{"module": name, "size": len(nodes)} for name, nodes in mods]
            global_spec, mod_spec_rows = laplacian_spectrum_metrics(S, mods)
            for r in rows:
                ms = next((m for m in mod_spec_rows if m["module"] == r["module"]), None)
                if ms: r.update(ms)
            pd.DataFrame(rows).to_csv(ep_dir / f"modules_L{L}.csv", index=False)

            # ---- Baselines: SVCCA / PWCCA / CKA drift ----
            svcca = pwcca = float("nan")
            CKA = float("nan")
            if prev["A"][L] is not None:
                try:
                    svcca, pwcca = svcca_pwcca(prev["A"][L], H, var_keep=0.99, max_dim=128)
                except Exception:
                    pass
                try:
                    CKA = cka_linear(prev["A"][L], H)
                except Exception:
                    pass

            # ---- Hungarian continuity (turnover) ----
            turnover_hung, matches = (float("nan"), [])
            if prev["modules"][L]:
                turnover_hung, matches = continuity_hungarian(prev["modules"][L], mods, prev["S"][L], S)
                pd.DataFrame(matches, columns=["prev","curr","cos"]).to_csv(ep_dir / f"match_hungarian_L{L}.csv", index=False)

            # ---- Nulls (degree-preserving rewires) + bootstrap CI ----
            mean_null = lo_null = hi_null = float("nan")
            if N_NULL_BOOTSTRAPS > 0 and prev["modules"][L]:
                null_vals = continuity_nulls(prev["modules"][L], prev["S"][L], S, B=N_NULL_BOOTSTRAPS, rng=args.seed+epoch+L)
                if len(null_vals):
                    mean_null, (lo_null, hi_null) = bootstrap_ci(np.array(null_vals, dtype=np.float32), alpha=0.05, B=min(200, 2*N_NULL_BOOTSTRAPS))

            # ---- Scalars ----
            Hrepr, PR = pca_entropy_and_pr(H, max_r=128)
            try:
                part = partition_with_singletons(G, mods)
                Q = float(nx.algorithms.community.quality.modularity(G, part, weight="weight"))
            except Exception:
                Q = 0.0
            Hdeg = degree_entropy(G)
            vel = float(1.0 - CKA) if np.isfinite(CKA) else np.nan

            layer_metrics = {
                "layer": L,
                "modularity_Q": Q,
                "deg_entropy": Hdeg,
                "repr_entropy": Hrepr,
                "participation_ratio": PR,
                "cka_prev": float(CKA) if np.isfinite(CKA) else np.nan,
                "svcca_prev": float(svcca) if np.isfinite(svcca) else np.nan,
                "pwcca_prev": float(pwcca) if np.isfinite(pwcca) else np.nan,
                "velocity": vel,
                "turnover_hungarian": float(turnover_hung) if np.isfinite(turnover_hung) else np.nan,
                "null_turnover_mean": float(mean_null),
                "null_turnover_ci_lo": float(lo_null),
                "null_turnover_ci_hi": float(hi_null),
                "edges": G.number_of_edges(),
                "nodes": G.number_of_nodes(),
                "topk": args.graph_topk
            }
            layer_metrics.update(global_spec)
            (ep_dir / f"metrics_L{L}.json").write_text(json.dumps(layer_metrics, indent=2))
            m_all = dict(base_metrics); m_all.update(layer_metrics)
            metrics_rows.append(m_all)

            # ---- Top-module attention PNG (one per layer per epoch) ----
            try:
                h_single = actlog.buffers[L][-1].numpy() if len(actlog.buffers[L]) > 0 else None
                if h_single is not None:
                    save_top_module_attention_map(
                        model, ref_cache, mods, h_single.astype(np.float32, copy=False),
                        out_png=str(ep_dir / f"layer_L{L}_topmodule.png")
                    )
            except Exception as e:
                (ep_dir / f"layer_L{L}_topmodule.err.txt").write_text(str(e))

            # ---- Update prevs / history ----
            prev["modules"][L] = mods
            prev["S"][L] = S
            prev["A"][L] = H
            prev["hist_mods"][L].append((mods, S))

        # Optional patch-level visuals / deltas
        if args.save_spatial:
            vis_dir = ep_dir / "spatial"; vis_dir.mkdir(exist_ok=True)
            kdone = 0
            for (Xk, yk, cid, sdir) in ref_cache[:args.spatial_examples]:
                Xk = Xk.to(device)
                y_scalar = float(yk.view(-1).cpu().numpy()) if yk is not None else None
                y_pred, w, s = gradcam_patch_attribution(model, Xk)
                contact_sheet_with_heat(Xk.detach().cpu(), s.detach().cpu(), str(vis_dir / f"{cid}_{sdir}_heat.png"), order_by="score", cols=8)
                y0, y1, dlogit, _ = delta_pred_and_auc(model, Xk, y_scalar, s, top_frac=args.attr_top_frac, auc_threshold=args.auc_threshold)
                pd.DataFrame([{"case_id": cid, "slide": sdir, "y0": y0, "y1": y1, "delta_pred": dlogit, "top_frac": args.attr_top_frac}]).to_csv(vis_dir / f"{cid}_{sdir}_delta.csv", index=False)
                kdone += 1
                if kdone >= args.spatial_examples: break

        dash_png = str(ep_dir / "global_epoch.png")
        plot_epoch_dashboard(layer_pngs, dash_png, epoch, {"train_loss": tr_loss, "val_loss": va_loss})

        # GC
        gc.collect()
        try: torch.cuda.empty_cache()
        except Exception: pass

    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(Path(args.out_dir) / "metrics_all_layers.csv", index=False)
    actlog.remove()
    with open(log_path, "a") as f: f.write("run_end\n")
    print("Done.")

# --- entrypoint guard ---
if __name__ == "__main__":
    import traceback, sys
    print("[entry] machine3_best.py launching main()", flush=True)
    try:
        main()
        print("[entry] main() finished successfully", flush=True)
    except SystemExit:
        raise
    except Exception as e:
        print("[entry] exception:", e, flush=True)
        traceback.print_exc(file=sys.stdout)
        raise
