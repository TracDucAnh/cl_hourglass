"""Bootstrap theo cụm nội dung và kiểm định hoán vị.

Đơn vị lấy mẫu lại là CỤM: cùng một nội dung c ở mọi ngôn ngữ (và mọi cặp ngôn ngữ) là một đơn vị, vì các
quan sát trong cùng cụm không độc lập (quy ước đo của kế hoạch).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np


@dataclass
class BootstrapResult:
    estimate: object
    ci_low: object
    ci_high: object
    se: object
    n_boot: int
    alpha: float
    boots: Optional[np.ndarray] = field(default=None, repr=False)


def cluster_indices(clusters: Sequence) -> List[np.ndarray]:
    """Danh sách chỉ số hàng của từng cụm (thứ tự theo giá trị cụm đã sắp xếp)."""
    clusters = np.asarray(clusters)
    order = np.argsort(clusters, kind="stable")
    sorted_c = clusters[order]
    _, start = np.unique(sorted_c, return_index=True)
    ends = list(start[1:]) + [len(order)]
    return [order[s:e] for s, e in zip(start, ends)]


def cluster_bootstrap(
    stat_fn: Callable,
    *arrays,
    clusters: Optional[Sequence] = None,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    return_boots: bool = False,
) -> BootstrapResult:
    """Bootstrap phần trăm cho stat_fn(*arrays) với lấy mẫu lại theo cụm (có hoàn lại).

    arrays: các mảng cùng số hàng (trục 0); stat_fn nhận các mảng con cùng thứ tự và trả về số hoặc vector.
    clusters: nhãn cụm cho mỗi hàng; None = mỗi hàng là một cụm (bootstrap thường).
    Lưu ý với số đo dựa trên ghép cặp (như G_k): cụm lặp lại tạo ra các hàng trùng, đó là hành vi mong đợi của
    bootstrap cụm; hãy chọn cụm là id nội dung chứ không phải id (nội dung, ngôn ngữ).
    """
    if not arrays:
        raise ValueError("cần ít nhất một mảng")
    arrays = tuple(np.asarray(a) for a in arrays)
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("các mảng phải cùng số hàng")
    groups = cluster_indices(np.arange(n) if clusters is None else clusters)
    if len(groups) < 2:
        raise ValueError("cần ít nhất 2 cụm")
    est = np.asarray(stat_fn(*arrays), dtype=np.float64)
    rng = np.random.default_rng(seed)
    G = len(groups)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, G, size=G)
        idx = np.concatenate([groups[g] for g in pick])
        boots.append(np.asarray(stat_fn(*(a[idx] for a in arrays)), dtype=np.float64))
    boots = np.stack(boots)
    lo = np.nanpercentile(boots, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(boots, 100 * (1 - alpha / 2), axis=0)
    se = np.nanstd(boots, axis=0, ddof=1)
    f = (lambda x: float(x)) if est.ndim == 0 else (lambda x: x)
    return BootstrapResult(f(est), f(lo), f(hi), f(se), n_boot, alpha, boots if return_boots else None)


@dataclass
class PermResult:
    observed: float
    null: np.ndarray = field(repr=False)
    p_greater: float
    p_less: float
    p_two: float
    alternative: str

    @property
    def p(self) -> float:
        return {"greater": self.p_greater, "less": self.p_less, "two-sided": self.p_two}[self.alternative]


def perm_null(
    stat_fn: Callable,
    A,
    B=None,
    *,
    n_perm: int = 1000,
    seed: int = 0,
    mode: str = "pairing",
    alternative: str = "greater",
) -> PermResult:
    """Phân phối rỗng bằng hoán vị.

    mode='pairing': xáo trộn hàng của B rồi tính stat_fn(A, B_hoán_vị) (ghép cặp bị xáo trộn của kế hoạch).
    mode='sign'   : A là vector hiệu cặp; đổi dấu ngẫu nhiên từng phần tử rồi tính stat_fn(A_đổi_dấu).
    p-value có hiệu chỉnh +1: p = (1 + #{null >= obs}) / (n_perm + 1).
    """
    if alternative not in ("greater", "less", "two-sided"):
        raise ValueError("alternative phải là 'greater', 'less' hoặc 'two-sided'")
    A = np.asarray(A)
    rng = np.random.default_rng(seed)
    if mode == "pairing":
        if B is None:
            raise ValueError("mode='pairing' cần B")
        B = np.asarray(B)
        if len(A) != len(B):
            raise ValueError("A và B phải cùng số hàng")
        obs = float(stat_fn(A, B))
        null = np.array([stat_fn(A, B[rng.permutation(len(B))]) for _ in range(n_perm)], dtype=np.float64)
    elif mode == "sign":
        obs = float(stat_fn(A))
        null = np.array([stat_fn(A * rng.choice([-1.0, 1.0], size=len(A))) for _ in range(n_perm)],
                        dtype=np.float64)
    else:
        raise ValueError("mode phải là 'pairing' hoặc 'sign'")
    pg = (1 + int((null >= obs).sum())) / (n_perm + 1)
    pl = (1 + int((null <= obs).sum())) / (n_perm + 1)
    p2 = min(1.0, 2 * min(pg, pl))
    return PermResult(obs, null, pg, pl, p2, alternative)