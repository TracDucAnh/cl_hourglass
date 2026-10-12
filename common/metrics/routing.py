"""Độ phân kỳ định tuyến của MoE (tương đương D_k cho MoE; Bảng 5, giao thức của Bandarkar et al. 2026).

Với mỗi câu và mỗi tầng MoE, phân phối định tuyến P[i, k, :] là tần suất mỗi expert được chọn (top-k) trên
các token của câu (chuẩn hóa về tổng 1). Độ phân kỳ của một ngôn ngữ so với một ngôn ngữ khác (thường là
tiếng Anh) ở tầng k là JS(P_A[i, k], P_B[i, k]) trung bình trên các câu song song i.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def normalize_dist(P, axis: int = -1) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    s = P.sum(axis=axis, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(s > 0, P / s, np.nan)
    return out


def js_divergence(P, Q, base: float = 2.0, axis: int = -1) -> np.ndarray:
    """JS(P, Q) = 0.5 KL(P||M) + 0.5 KL(Q||M), M = (P+Q)/2; quy ước 0 log 0 = 0.

    base=2: nằm trong [0, 1] (bit). Hàng có tổng 0 (tầng dense không có router) cho NaN.
    """
    P, Q = normalize_dist(P, axis), normalize_dist(Q, axis)
    M = 0.5 * (P + Q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = np.where(P > 0, P * (np.log(P) - np.log(M)), 0.0)
        kl_qm = np.where(Q > 0, Q * (np.log(Q) - np.log(M)), 0.0)
    js = 0.5 * (kl_pm.sum(axis=axis) + kl_qm.sum(axis=axis)) / np.log(base)
    nan_mask = np.isnan(P).any(axis=axis) | np.isnan(Q).any(axis=axis)
    return np.where(nan_mask, np.nan, np.clip(js, 0.0, None))


def js_routing(PA, PB, base: float = 2.0, return_items: bool = False) -> np.ndarray:
    """JS định tuyến theo tầng, trung bình trên các câu.

    PA, PB: [n_items, n_layers, n_experts] (hàng i của PA và PB là cùng một nội dung).
    Trả về vector [n_layers]; return_items=True trả về [n_items, n_layers].
    """
    PA, PB = np.asarray(PA), np.asarray(PB)
    if PA.shape != PB.shape or PA.ndim != 3:
        raise ValueError(f"PA {PA.shape} và PB {PB.shape} phải cùng kích thước [n_items, n_layers, n_experts]")
    js = js_divergence(PA, PB, base=base, axis=-1)
    if return_items:
        return js
    with np.errstate(all="ignore"):
        return np.nanmean(js, axis=0)


def expert_usage(topk_idx: np.ndarray, n_experts: int, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Tần suất chọn expert của MỘT câu ở MỘT tầng.

    topk_idx: [n_tokens, top_k] chỉ số expert được chọn cho mỗi token; mask: [n_tokens] bool (token được tính).
    Trả về vector [n_experts] chuẩn hóa tổng 1 (toàn 0 nếu không có token nào).
    """
    topk_idx = np.asarray(topk_idx)
    if mask is not None:
        topk_idx = topk_idx[np.asarray(mask, dtype=bool)]
    cnt = np.bincount(topk_idx.reshape(-1), minlength=n_experts).astype(np.float64)
    s = cnt.sum()
    return cnt / s if s > 0 else cnt