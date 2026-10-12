"""Chạy cả bộ đại lượng đo của Bảng 5 trên một tensor biểu diễn [n_items, L, K+1, d] và trả về bảng
dạng dài đúng schema "curves" của common/io.py (cột: family, model, dataset, lang_a, lang_b, layer, rho,
metric, value).

Tên đại lượng: D (Dk), G (Gk), CKA, PROC (Procrustes kiểm chéo), R1 (acc@1 truy hồi CSLS), MRR.
Uk (usable information) và retrieval theo từ không nằm ở đây: Uk theo từng ngôn ngữ chứ không theo cặp.
"""
from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .alignment import G_k, D_std, cka_unbiased, pooled_scale, procrustes_cv, retrieval_csls
from .curves import curve_features, has_waist

ALL_METRICS = ("D", "G", "CKA", "PROC", "R1", "MRR")


def _pair_seed(seed: int, layer: int, pair_idx: int) -> int:
    return int((seed * 1_000_003 + layer * 10_007 + pair_idx) % (2**32))


def language_pairs(langs: Sequence[str]) -> List[Tuple[str, str]]:
    return list(itertools.combinations(langs, 2))


def compute_curves(
    Z,
    langs: Sequence[str],
    *,
    metrics: Sequence[str] = ALL_METRICS,
    pairs: Optional[Sequence[Tuple[str, str]]] = None,
    layers: Optional[Sequence[int]] = None,
    n_perm: int = 200,
    d_pca: Optional[int] = 64,
    csls_k: int = 10,
    seed: int = 0,
    standardize: bool = True,
    family: str = "",
    model: str = "",
    dataset: str = "",
    progress: bool = False,
) -> pd.DataFrame:
    """Z: [n_items, L, K+1, d] (ndarray hoặc memmap, fp16 được). langs: tên L ngôn ngữ theo trục 1.

    Mỗi tầng được trừ trung bình theo ngôn ngữ và chia cho độ lệch chuẩn từng chiều tính chung trên mọi
    ngôn ngữ, nên Dk, Gk của các cặp so sánh được với nhau. CKA, Procrustes và truy hồi dùng cùng dữ liệu đó
    (CKA không chuẩn hóa từng chiều; Procrustes chuẩn hóa sau PCA; retrieval dùng cosine, xem alignment.py).
    """
    bad = [m for m in metrics if m not in ALL_METRICS]
    if bad:
        raise ValueError(f"đại lượng không biết: {bad}; chọn trong {ALL_METRICS}")
    Z = Z if hasattr(Z, "shape") else np.asarray(Z)
    if Z.ndim != 4:
        raise ValueError(f"Z phải có dạng [n_items, L, K+1, d], nhận {Z.shape}")
    n, L, nk, d = Z.shape
    if len(langs) != L:
        raise ValueError(f"len(langs)={len(langs)} nhưng Z có {L} ngôn ngữ")
    K = nk - 1
    lang_idx = {l: i for i, l in enumerate(langs)}
    pairs = list(pairs) if pairs is not None else language_pairs(langs)
    layer_list = list(range(nk)) if layers is None else list(layers)
    want = set(metrics)

    rows: List[dict] = []
    it = layer_list
    if progress:
        try:
            from tqdm import tqdm

            it = tqdm(layer_list, desc="layers")
        except ImportError:
            pass
    for k in it:
        X = np.asarray(Z[:, :, k, :], dtype=np.float32)          # [n, L, d]
        X = X - X.mean(0, keepdims=True)                          # trừ trung bình từng ngôn ngữ
        scale = pooled_scale(X, center=False) if standardize else None
        for pi, (a, b) in enumerate(pairs):
            A, B = X[:, lang_idx[a], :], X[:, lang_idx[b], :]
            s = _pair_seed(seed, k, pi)
            vals: Dict[str, float] = {}
            if "D" in want:
                vals["D"] = D_std(A, B, center=False, standardize=standardize, scale=scale)
            if "G" in want:
                vals["G"] = G_k(A, B, n_perm=n_perm, seed=s, center=False, standardize=standardize, scale=scale)
            if "CKA" in want:
                vals["CKA"] = cka_unbiased(A, B, center=False)
            if "PROC" in want:
                vals["PROC"] = procrustes_cv(A, B, d_pca=d_pca, seed=s)
            if want & {"R1", "MRR"}:
                r = retrieval_csls(A, B, k=csls_k, center=False, standardize=False)
                vals["R1"], vals["MRR"] = r["acc1"], r["mrr"]
            for m, v in vals.items():
                if m in want:
                    rows.append({
                        "family": family, "model": model, "dataset": dataset, "lang_a": a, "lang_b": b,
                        "layer": int(k), "rho": float(k / K) if K > 0 else 0.0, "metric": m, "value": float(v),
                    })
    return pd.DataFrame(rows)


def mean_curve(curves: pd.DataFrame, metric: str, pairs: Optional[Iterable[Tuple[str, str]]] = None) -> pd.DataFrame:
    """Trung bình theo cặp ngôn ngữ ở từng tầng: cột layer, rho, mean, std, n_pairs."""
    df = curves[curves["metric"] == metric]
    if pairs is not None:
        keep = set(pairs)
        df = df[[(a, b) in keep for a, b in zip(df["lang_a"], df["lang_b"])]]
    g = df.groupby(["layer", "rho"])["value"]
    out = g.agg(mean="mean", std="std", n_pairs="count").reset_index().sort_values("layer")
    return out.reset_index(drop=True)


def features_from_curves(curves: pd.DataFrame, theta: float = 0.25, main: str = "G", consensus: str = "CKA",
                         tol: int = 1) -> pd.DataFrame:
    """Đặc trưng đường cong (k*, rho*, w, S) cho từng cặp và cho trung bình mọi cặp (lang_a = lang_b = 'ALL').

    Bảng đúng schema "features" của io.py, thêm cột has_waist (quy tắc đồng thuận G_k với CKA, nếu CKA có
    trong `curves`) và g_max, baseline.
    """
    meta_cols = ["family", "model", "dataset"]
    first = curves.iloc[0]
    meta = {c: first[c] for c in meta_cols}
    have_cons = consensus in set(curves["metric"])
    rows = []

    def one(a: str, b: str, G: np.ndarray, C: Optional[np.ndarray]):
        f = curve_features(G, theta=theta)
        hw = has_waist(G, C, tol=tol) if C is not None else has_waist(G)
        rows.append({**meta, "lang_a": a, "lang_b": b, "metric": main, "k_star": f["k_star"],
                     "rho_star": f["rho_star"], "width": f["width"], "width_rel": f["width_rel"],
                     "amplitude": f["amplitude"], "g_max": f["g_max"], "baseline": f["baseline"],
                     "has_waist": bool(hw)})

    sub = curves[curves["metric"] == main]
    for (a, b), g in sub.groupby(["lang_a", "lang_b"]):
        G = g.sort_values("layer")["value"].to_numpy()
        C = None
        if have_cons:
            c = curves[(curves["metric"] == consensus) & (curves["lang_a"] == a) & (curves["lang_b"] == b)]
            C = c.sort_values("layer")["value"].to_numpy()
        one(a, b, G, C)
    Gm = mean_curve(curves, main)["mean"].to_numpy()
    Cm = mean_curve(curves, consensus)["mean"].to_numpy() if have_cons else None
    one("ALL", "ALL", Gm, Cm)
    return pd.DataFrame(rows)