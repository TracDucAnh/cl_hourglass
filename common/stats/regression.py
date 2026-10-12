"""Hồi quy đoạn và "hai đường thẳng" cho đường cong theo tầng (P1) và đường cong chữ U (S3, S5).

segmented_fit: hồi quy tuyến tính từng đoạn LIÊN TỤC với 1 hoặc 2 điểm gãy, tìm điểm gãy bằng lưới (tối thiểu hóa RSS).
two_lines    : kiểm định hình chữ U / U ngược bằng hai đường thẳng riêng ở hai phía của một điểm gãy
               (tinh thần của Simonsohn 2018, "Two-lines test").

Lưu ý thống kê quan trọng:
  * p-value F thông thường của mô hình có điểm gãy tìm từ dữ liệu là SAI (điểm gãy là tham số nuisance chỉ có
    dưới H1). Dùng `n_boot > 0` để lấy p-value bootstrap dư (residual bootstrap) dưới H0 "một đường thẳng".
  * two_lines ở đây chọn điểm gãy bằng segmented_fit rồi kiểm trên cùng dữ liệu, nên hơi quá lạc quan
    (Simonsohn dùng thuật toán "Robin Hood" để tránh); nếu cần kết luận chặt, chia đôi dữ liệu (chọn điểm gãy ở
    nửa này, kiểm ở nửa kia) hoặc truyền `xb` cố định đăng ký trước.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy import stats as sps


# ------------------------------------------------------------------ tiện ích
def _design(x: np.ndarray, breaks: Sequence[float]) -> np.ndarray:
    cols = [np.ones_like(x), x] + [np.maximum(x - b, 0.0) for b in breaks]
    return np.column_stack(cols)


def _lstsq_rss(X: np.ndarray, y: np.ndarray):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ coef
    return coef, float(r @ r)


@dataclass
class SegmentedFit:
    breaks: np.ndarray
    coef: np.ndarray
    slopes: np.ndarray
    rss: float
    rss_linear: float
    n: int
    n_params: int
    bic: float
    bic_linear: float
    f_stat: float
    p_value: Optional[float] = None
    n_boot: int = 0
    _breaks_list: list = field(default_factory=list, repr=False)

    @property
    def delta_bic(self) -> float:
        """bic_tuyến_tính - bic_đoạn (dương: mô hình có điểm gãy tốt hơn)."""
        return self.bic_linear - self.bic

    def predict(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return _design(x, self.breaks) @ self.coef


def _bic(n: int, rss: float, k: int) -> float:
    return n * np.log(max(rss, 1e-300) / n) + k * np.log(n)


def _grid_search(x, y, n_breaks, min_seg, n_grid):
    xs = np.unique(x)
    if len(xs) < 2 * min_seg + 1:
        raise ValueError(f"cần ít nhất {2 * min_seg + 1} giá trị x khác nhau, có {len(xs)}")
    cand = np.linspace(xs[min_seg - 1], xs[-min_seg], n_grid)
    best = (np.inf, None, None)
    if n_breaks == 1:
        for b in cand:
            X = _design(x, [b])
            coef, rss = _lstsq_rss(X, y)
            if rss < best[0]:
                best = (rss, [b], coef)
    elif n_breaks == 2:
        for i, b1 in enumerate(cand):
            for b2 in cand[i + 1:]:
                if ((x > b1) & (x <= b2)).sum() < 2:
                    continue
                X = _design(x, [b1, b2])
                coef, rss = _lstsq_rss(X, y)
                if rss < best[0]:
                    best = (rss, [b1, b2], coef)
    else:
        raise ValueError("n_breaks phải là 1 hoặc 2")
    if best[1] is None:
        raise ValueError("không tìm được điểm gãy hợp lệ")
    return best


def segmented_fit(x, y, n_breaks: int = 1, min_seg: int = 3, n_grid: int = 200, n_boot: int = 0, seed: int = 0):
    """Hồi quy tuyến tính từng đoạn liên tục; trả về SegmentedFit.

    min_seg: số giá trị x khác nhau tối thiểu ở mỗi đầu. n_grid: số điểm gãy ứng viên (với 2 điểm gãy
    nên dùng n_grid <= 60). n_boot > 0: p-value bootstrap dư dưới H0 một đường thẳng.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x, y phải là vector cùng độ dài")
    n = len(x)
    rss, breaks, coef = _grid_search(x, y, n_breaks, min_seg, n_grid)
    X0 = _design(x, [])
    c0, rss0 = _lstsq_rss(X0, y)
    p_seg = 2 + n_breaks + n_breaks            # hệ số hồi quy + các điểm gãy (tham số tự do)
    p_lin = 2
    df2 = n - (2 + n_breaks)
    f = ((rss0 - rss) / n_breaks) / (rss / max(df2, 1)) if rss > 0 else np.inf
    slopes = np.cumsum(coef[1:])               # độ dốc của từng đoạn
    fit = SegmentedFit(np.array(breaks), coef, slopes, rss, rss0, n, p_seg, _bic(n, rss, p_seg),
                       _bic(n, rss0, p_lin), float(f))
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        resid = y - X0 @ c0
        fitted0 = X0 @ c0
        ge = 0
        ng = min(n_grid, 60)
        for _ in range(n_boot):
            yb = fitted0 + rng.choice(resid, size=n, replace=True)
            r_b, _, _ = _grid_search(x, yb, n_breaks, min_seg, ng)
            _, r0_b = _lstsq_rss(X0, yb)
            fb = ((r0_b - r_b) / n_breaks) / (r_b / max(df2, 1)) if r_b > 0 else np.inf
            ge += fb >= f
        fit.p_value = (1 + ge) / (n_boot + 1)
        fit.n_boot = n_boot
    return fit


# ------------------------------------------------------------------ hai đường thẳng
def _ols_line(x: np.ndarray, y: np.ndarray) -> dict:
    n = len(x)
    if n < 3:
        return {"n": n, "slope": float("nan"), "se": float("nan"), "t": float("nan"), "p": float("nan")}
    X = np.column_stack([np.ones(n), x])
    coef, rss = _lstsq_rss(X, y)
    s2 = rss / (n - 2)
    xtx_inv = np.linalg.inv(X.T @ X)
    se = float(np.sqrt(s2 * xtx_inv[1, 1]))
    t = float(coef[1] / se) if se > 0 else float("inf") * np.sign(coef[1])
    p = float(2 * sps.t.sf(abs(t), n - 2))
    return {"n": n, "slope": float(coef[1]), "se": se, "t": t, "p": p}


def two_lines(x, y, xb: Optional[float] = None, min_seg: int = 3, alpha: float = 0.05) -> dict:
    """Hai đường thẳng riêng ở hai phía điểm gãy xb (mặc định: điểm gãy của segmented_fit).

    shape = 'inverted_U' nếu độ dốc trái > 0 và độ dốc phải < 0, đều có ý nghĩa ở mức alpha (kiểm hai phía
    của từng độ dốc); 'U' nếu ngược lại; 'none' nếu không. Trả về dict (xb, left, right, shape).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if xb is None:
        xb = float(segmented_fit(x, y, n_breaks=1, min_seg=min_seg).breaks[0])
    L = _ols_line(x[x <= xb], y[x <= xb])
    R = _ols_line(x[x > xb], y[x > xb])
    shape = "none"
    if L["p"] < alpha and R["p"] < alpha:
        if L["slope"] > 0 and R["slope"] < 0:
            shape = "inverted_U"
        elif L["slope"] < 0 and R["slope"] > 0:
            shape = "U"
    return {"xb": float(xb), "left": L, "right": R, "shape": shape, "alpha": alpha}