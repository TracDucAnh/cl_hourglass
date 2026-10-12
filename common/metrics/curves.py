"""Đặc trưng của đường cong theo tầng (k*, rho*, w, S) và quy tắc "có đáy" của kế hoạch.

Đường G_k theo tầng k = 0..K có dạng chuông ngược của hourglass: thấp ở hai đầu (các ngôn ngữ còn tách),
cao ở eo (các ngôn ngữ co lại). Quy ước (kế hoạch, mục 2 "Quy ước đo"):
  k*        = argmax_k G_k            (lấy tầng đầu tiên nếu có nhiều tầng bằng nhau)
  rho*      = k* / K                  (độ sâu tương đối, K = chỉ số tầng cuối)
  baseline  = (G_0 + G_K) / 2         (mức trung bình của hai đầu)
  S         = G_{k*} - baseline       (biên độ)
  w         = độ rộng, tính bằng tầng, của đoạn liên tục quanh k* nơi G >= baseline + theta * S
              (theta = 0.25; nội suy tuyến tính tại hai giao điểm); w_rel = w / K
"có đáy" chỉ khi k* theo G_k và k* theo CKA lệch nhau <= 1 tầng (và k* nằm ở trong, không ở hai đầu).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def curve_features(G, theta: float = 0.25) -> Dict[str, float]:
    G = np.asarray(G, dtype=np.float64)
    if G.ndim != 1 or len(G) < 3:
        raise ValueError("G phải là vector theo tầng, ít nhất 3 tầng")
    if np.all(np.isnan(G)):
        raise ValueError("G toàn NaN")
    K = len(G) - 1
    k_star = int(np.nanargmax(G))
    g_max = float(G[k_star])
    baseline = float(0.5 * (G[0] + G[K]))
    S = g_max - baseline
    out = {
        "k_star": float(k_star), "rho_star": k_star / K, "amplitude": S, "g_max": g_max,
        "baseline": baseline, "K": float(K), "width": float("nan"), "width_rel": float("nan"),
        "left": float("nan"), "right": float("nan"),
    }
    if not S > 0:
        return out
    thr = baseline + theta * S
    # đi sang trái
    j = k_star
    while j > 0 and G[j - 1] >= thr:
        j -= 1
    if j == 0:
        left = 0.0
    else:                                   # G[j-1] < thr <= G[j]
        left = (j - 1) + (thr - G[j - 1]) / (G[j] - G[j - 1])
    # đi sang phải
    j = k_star
    while j < K and G[j + 1] >= thr:
        j += 1
    if j == K:
        right = float(K)
    else:                                   # G[j] >= thr > G[j+1]
        right = j + (G[j] - thr) / (G[j] - G[j + 1])
    out.update(left=float(left), right=float(right), width=float(right - left), width_rel=float((right - left) / K))
    return out


def has_waist(G, cka=None, tol: int = 1, require_interior: bool = True, min_amplitude: float = 0.0,
              return_info: bool = False):
    """Quy tắc đồng thuận của kế hoạch: k* theo G_k và theo CKA lệch <= tol tầng.

    cka=None: chỉ dùng G_k (không kiểm đồng thuận). require_interior: k* không được là tầng 0 hay tầng K.
    """
    G = np.asarray(G, dtype=np.float64)
    kG = int(np.nanargmax(G))
    K = len(G) - 1
    feats = curve_features(G)
    ok = feats["amplitude"] > min_amplitude
    if require_interior:
        ok = ok and 0 < kG < K
    kC = None
    gap = None
    if cka is not None:
        cka = np.asarray(cka, dtype=np.float64)
        if len(cka) != len(G):
            raise ValueError("G và CKA phải cùng số tầng")
        kC = int(np.nanargmax(cka))
        gap = abs(kG - kC)
        ok = ok and gap <= tol
    if return_info:
        return {"has_waist": bool(ok), "k_star_G": kG, "k_star_CKA": kC, "gap": gap, "amplitude": feats["amplitude"]}
    return bool(ok)