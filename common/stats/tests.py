"""Kiểm định cho so sánh có cặp (mục 5 của kế hoạch).

  wilcoxon_exact  Wilcoxon hạng có dấu, p-value CHÍNH XÁC (kể cả khi có hạng đồng hạng, nhờ quy hoạch động trên
                  hạng nhân đôi), kèm ước lượng Hodges-Lehmann và khoảng tin cậy
  tost            hai kiểm định một phía cho tương đương ("không đổi" kiểm bằng biên = SESOI)
  holm, benjamini_hochberg  hiệu chỉnh đa kiểm định (Holm trong từng giả thuyết, BH cho khám phá)

Với 6 seed, p hai phía nhỏ nhất của Wilcoxon chính xác là 2/64 = 0.03125 (kế hoạch ghi 0,031).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from scipy import stats as sps


# ------------------------------------------------------------------ Wilcoxon chính xác
def _signed_rank_pmf(ranks2: np.ndarray) -> np.ndarray:
    """pmf của 2*W+ (W+ = tổng hạng của các phần tử dương) dưới H0, khi mỗi hạng vào tổng với xác suất 1/2."""
    total = int(ranks2.sum())
    dp = np.zeros(total + 1)
    dp[0] = 1.0
    for r in ranks2:
        new = 0.5 * dp
        new[r:] += 0.5 * dp[: total + 1 - r]
        dp = new
    return dp


def wilcoxon_exact(x, y=None, alternative: str = "two-sided", ci_level: float = 0.95) -> Dict:
    """Wilcoxon hạng có dấu chính xác cho hiệu cặp d = x - y (hoặc x nếu y=None).

    Số 0 bị bỏ (zero_method='wilcox'). Hạng đồng hạng dùng hạng trung bình và phân phối chính xác có điều kiện
    trên các hạng đó (không dùng xấp xỉ chuẩn).
    alternative: 'two-sided' | 'greater' (x > y) | 'less'.
    Trả về dict: statistic (W+), p, n (số hiệu khác 0), hl (ước lượng Hodges-Lehmann = trung vị các trung bình
    Walsh), hl_ci (khoảng tin cậy, chính xác khi không có đồng hạng), hl_ci_level (mức phủ thực tế).
    """
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative phải là 'two-sided', 'greater' hoặc 'less'")
    d = np.asarray(x, dtype=np.float64)
    if y is not None:
        d = d - np.asarray(y, dtype=np.float64)
    d = d[~np.isnan(d)]
    d = d[d != 0]
    n = len(d)
    out = {"n": n, "alternative": alternative}
    if n == 0:
        out.update(statistic=float("nan"), p=1.0, hl=float("nan"), hl_ci=(float("nan"), float("nan")), hl_ci_level=float("nan"))
        return out
    ranks = sps.rankdata(np.abs(d))
    ranks2 = np.rint(2 * ranks).astype(np.int64)
    w_plus = float(ranks[d > 0].sum())
    w2 = int(np.rint(2 * w_plus))
    pmf = _signed_rank_pmf(ranks2)
    p_greater = float(pmf[w2:].sum())
    p_less = float(pmf[: w2 + 1].sum())
    p = {"greater": p_greater, "less": p_less, "two-sided": min(1.0, 2 * min(p_greater, p_less))}[alternative]

    # Hodges-Lehmann
    iu = np.triu_indices(n)
    walsh = np.sort(((d[:, None] + d[None, :]) / 2.0)[iu])
    hl = float(np.median(walsh))
    pmf0 = _signed_rank_pmf(2 * np.arange(1, n + 1))          # không đồng hạng: hạng 1..n
    cdf = np.cumsum(pmf0)                                      # cdf[j] = P(2W+ <= j)
    cdf_w = cdf[::2]                                           # P(W+ <= k) = cdf[2k]
    alpha = 1 - ci_level
    ok = np.where(cdf_w <= alpha / 2)[0]
    t = int(ok.max()) if len(ok) else 0
    M = len(walsh)
    lo, hi = walsh[t], walsh[M - 1 - t]
    out.update(statistic=w_plus, p=p, p_greater=p_greater, p_less=p_less, hl=hl,
               hl_ci=(float(lo), float(hi)), hl_ci_level=float(1 - 2 * cdf_w[t]))
    return out


# ------------------------------------------------------------------ TOST
def tost(x, y=None, margin: Optional[float] = None, *, low: Optional[float] = None, high: Optional[float] = None,
         paired: bool = True, alpha: float = 0.05) -> Dict:
    """Two One-Sided Tests cho tương đương: H0: |mean(x - y)| >= margin; H1: -margin < mean(x - y) < margin.

    y=None: x đã là các hiệu (một mẫu). paired=True với y: dùng hiệu cặp. paired=False: Welch hai mẫu.
    Biên: margin (đối xứng) hoặc (low, high). p = max(p_dưới, p_trên); tương đương nếu p < alpha, tức
    khoảng tin cậy 100(1-2*alpha)% nằm hoàn toàn trong (low, high).
    """
    if margin is not None:
        low, high = -abs(margin), abs(margin)
    if low is None or high is None or not low < high:
        raise ValueError("cần margin hoặc (low, high) với low < high")
    x = np.asarray(x, dtype=np.float64)
    if y is None or paired:
        d = x if y is None else x - np.asarray(y, dtype=np.float64)
        d = d[~np.isnan(d)]
        n = len(d)
        if n < 2:
            raise ValueError("cần ít nhất 2 quan sát")
        mean, se, df = float(d.mean()), float(d.std(ddof=1) / np.sqrt(n)), n - 1
    else:
        y = np.asarray(y, dtype=np.float64)
        v1, v2 = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
        mean, se = float(x.mean() - y.mean()), float(np.sqrt(v1 + v2))
        df = (v1 + v2) ** 2 / (v1**2 / (len(x) - 1) + v2**2 / (len(y) - 1))
    p_lower = float(sps.t.sf((mean - low) / se, df))      # H0: mean <= low
    p_upper = float(sps.t.cdf((mean - high) / se, df))    # H0: mean >= high
    p = max(p_lower, p_upper)
    tcrit = sps.t.ppf(1 - alpha, df)
    return {
        "mean_diff": mean, "se": se, "df": float(df), "p_lower": p_lower, "p_upper": p_upper, "p": p,
        "ci": (mean - tcrit * se, mean + tcrit * se), "ci_level": 1 - 2 * alpha,
        "equivalent": bool(p < alpha), "low": float(low), "high": float(high),
    }


# ------------------------------------------------------------------ đa kiểm định
def holm(pvals: Sequence[float]) -> np.ndarray:
    """p đã hiệu chỉnh theo Holm-Bonferroni (kiểm soát FWER), cùng thứ tự đầu vào."""
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def benjamini_hochberg(pvals: Sequence[float]) -> np.ndarray:
    """q-value theo Benjamini-Hochberg (kiểm soát FDR), cùng thứ tự đầu vào."""
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)[::-1]
    adj = np.empty(m)
    run = 1.0
    for k, i in enumerate(order):
        rank = m - k
        run = min(run, p[i] * m / rank)
        adj[i] = min(1.0, run)
    return adj