"""Số đo căn chỉnh giữa hai ngôn ngữ cho CÙNG nội dung (mục 3.5, 6.8, Bảng 5 của đề cương).

Quy ước chung: ZA, ZB là mảng [n, d]; hàng i của ZA và ZB là biểu diễn của cùng một nội dung c_i
ở hai ngôn ngữ. Mọi đại lượng đều tính "sau khi trừ trung bình theo ngôn ngữ" (center=True).

  D_std        khoảng cách chuẩn hóa Dk = E||ZA-ZB||^2 / (phương sai tổng)   (0: trùng, 2: độc lập)
  G_k          1 - D_true / mean(D_ghép_cặp_xáo_trộn)                         (1: trùng, 0: ngẫu nhiên)
  cka_unbiased CKA tuyến tính với HSIC không chệch (Song 2012), dạng ma trận Gram
  procrustes_cv  quay trực giao khớp trên nửa này, chấm điểm (cosine Frobenius) trên nửa kia
  retrieval_csls tìm câu tương đương bằng CSLS (giảm hubness)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

EPS = 1e-8


# ------------------------------------------------------------------ chuẩn bị
def _as2d(Z, dtype) -> np.ndarray:
    Z = np.asarray(Z)
    if Z.ndim != 2:
        raise ValueError(f"cần mảng 2 chiều [n, d], nhận {Z.shape}")
    return Z.astype(dtype, copy=True)


def pooled_scale(Zs, center: bool = True, eps: float = EPS) -> np.ndarray:
    """Độ lệch chuẩn từng chiều, gộp mọi ngôn ngữ (sau khi trừ trung bình từng ngôn ngữ).

    Zs: danh sách mảng [n, d] (một mảng mỗi ngôn ngữ) hoặc mảng [n, L, d].
    Dùng chung một `scale` cho mọi cặp ở cùng một tầng để các Dk/Gk so sánh được giữa các cặp.
    """
    if isinstance(Zs, np.ndarray) and Zs.ndim == 3:
        Zs = [Zs[:, j, :] for j in range(Zs.shape[1])]
    acc = 0.0
    cnt = 0
    for Z in Zs:
        Z = np.asarray(Z, dtype=np.float64)
        if center:
            Z = Z - Z.mean(0, keepdims=True)
        acc = acc + (Z**2).sum(0)
        cnt += Z.shape[0]
    return np.sqrt(acc / max(cnt, 1))


def prepare_pair(
    ZA,
    ZB,
    *,
    center: bool = True,
    standardize: bool = True,
    scale: Optional[np.ndarray] = None,
    eps: float = EPS,
    dtype=np.float64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Trừ trung bình theo ngôn ngữ, chia theo độ lệch chuẩn từng chiều, bỏ chiều suy biến."""
    A, B = _as2d(ZA, dtype), _as2d(ZB, dtype)
    if A.shape != B.shape:
        raise ValueError(f"ZA {A.shape} và ZB {B.shape} phải cùng kích thước")
    if center:
        A -= A.mean(0, keepdims=True)
        B -= B.mean(0, keepdims=True)
    if standardize:
        s = scale if scale is not None else pooled_scale([A, B], center=False)
        s = np.asarray(s, dtype=np.float64)
        keep = s > eps
        if not keep.all():
            A, B, s = A[:, keep], B[:, keep], s[keep]
        A /= s.astype(dtype)
        B /= s.astype(dtype)
    return A, B


# ------------------------------------------------------------------ D và G
def D_std(ZA, ZB, *, center: bool = True, standardize: bool = True,
          scale: Optional[np.ndarray] = None, dtype=np.float64) -> float:
    """Dk = E||ZA - ZB||^2 / sigma^2, sigma^2 = phương sai tổng của ZA, ZB gộp lại.

    Sau khi trừ trung bình từng ngôn ngữ: sigma^2 = (E||a||^2 + E||b||^2) / 2.
    Hai biểu diễn trùng nhau cho 0; hai biểu diễn độc lập (cùng phương sai) cho 2.
    """
    A, B = prepare_pair(ZA, ZB, center=center, standardize=standardize, scale=scale, dtype=dtype)
    num = float(((A - B) ** 2).sum(1).mean())
    var = 0.5 * float((A**2).sum(1).mean() + (B**2).sum(1).mean())
    return num / var if var > 0 else float("nan")


def _random_cycles(n: int, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """n_perm hoán vị không điểm bất động (mỗi hoán vị là một chu trình n phần tử ngẫu nhiên)."""
    out = np.empty((n_perm, n), dtype=np.int64)
    for p in range(n_perm):
        order = rng.permutation(n)
        out[p, order] = np.roll(order, -1)
    return out


def G_k(ZA, ZB, n_perm: int = 1000, seed: int = 0, *, center: bool = True, standardize: bool = True,
        scale: Optional[np.ndarray] = None, return_all: bool = False, dtype=np.float64):
    """Gk = 1 - D_true / mean(D_xáo_trộn).

    D_xáo_trộn: cùng công thức D nhưng ghép lại ZB theo hoán vị ngẫu nhiên không điểm bất động
    (cặp bị xáo trộn là mức ngẫu nhiên bắt buộc của kế hoạch). Dùng ma trận Gram M = A B^T tính một
    lần, nên mỗi hoán vị chỉ tốn O(n).

    return_all=True trả về dict {G, D_true, D_null_mean, D_null_std, p} với p = (1 + #{D_null <= D_true}) / (n_perm + 1).
    """
    A, B = prepare_pair(ZA, ZB, center=center, standardize=standardize, scale=scale, dtype=dtype)
    n = A.shape[0]
    if n < 3:
        raise ValueError("cần ít nhất 3 cặp để xáo trộn ghép cặp")
    na = (A**2).sum(1).mean()
    nb = (B**2).sum(1).mean()
    var = 0.5 * float(na + nb)
    d_true = float(((A - B) ** 2).sum(1).mean()) / var
    M = A @ B.T
    rng = np.random.default_rng(seed)
    perms = _random_cycles(n, n_perm, rng)
    idx = np.arange(n)
    cross = M[idx[None, :], perms].mean(1)          # mean_i a_i . b_pi(i)
    d_null = (na + nb - 2.0 * cross) / var
    g = 1.0 - d_true / float(d_null.mean())
    if not return_all:
        return float(g)
    return {
        "G": float(g), "D_true": d_true, "D_null_mean": float(d_null.mean()),
        "D_null_std": float(d_null.std(ddof=1)) if n_perm > 1 else float("nan"),
        "p": float((1 + (d_null <= d_true).sum()) / (n_perm + 1)),
    }


# ------------------------------------------------------------------ CKA không chệch
def _hsic_unbiased(K: np.ndarray, L: np.ndarray) -> float:
    """HSIC_1 của Song et al. (2012); K, L đối xứng n x n, đường chéo bị đặt 0."""
    n = K.shape[0]
    K = K.copy()
    L = L.copy()
    np.fill_diagonal(K, 0.0)
    np.fill_diagonal(L, 0.0)
    t1 = float((K * L).sum())
    t2 = float(K.sum()) * float(L.sum()) / ((n - 1) * (n - 2))
    t3 = 2.0 / (n - 2) * float(K.sum(0) @ L.sum(0))
    return (t1 + t2 - t3) / (n * (n - 3))


def cka_unbiased(ZA, ZB, *, center: bool = True, dtype=np.float64) -> float:
    """CKA tuyến tính, ước lượng HSIC không chệch trên ma trận Gram K = A A^T.

    Bất biến với phép quay và co giãn đẳng hướng: ZB = ZA R cho đúng 1.
    Không chuẩn hóa từng chiều (CKA tuyến tính vốn nhạy với chiều trội; Bảng 5 yêu cầu báo cáo cùng
    Procrustes để tránh phụ thuộc hướng trội).
    """
    A, B = _as2d(ZA, dtype), _as2d(ZB, dtype)
    if A.shape[0] != B.shape[0]:
        raise ValueError("ZA và ZB phải cùng số hàng")
    n = A.shape[0]
    if n < 4:
        raise ValueError("HSIC không chệch cần n >= 4")
    if center:
        A = A - A.mean(0, keepdims=True)
        B = B - B.mean(0, keepdims=True)
    K, L = A @ A.T, B @ B.T
    hkl, hkk, hll = _hsic_unbiased(K, L), _hsic_unbiased(K, K), _hsic_unbiased(L, L)
    den = np.sqrt(hkk * hll) if hkk > 0 and hll > 0 else 0.0
    return float(hkl / den) if den > 0 else float("nan")


# ------------------------------------------------------------------ Procrustes kiểm chéo
def _pca_fit(X: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(0, keepdims=True)
    # SVD kinh tế; hàng của Vt là các thành phần chính
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:p].T


def _procrustes_fit_score(a_tr, b_tr, a_te, b_te) -> float:
    """Khớp R trực giao trên tập khớp, chấm điểm trên tập thử bằng cosine Frobenius <a_te R, b_te> / (|a_te| |b_te|).

    Hai phía được chuẩn hóa Frobenius theo tập khớp (bỏ khác biệt thang chung). Điểm bằng 1 nếu b_te là bản
    quay của a_te và xấp xỉ 0 nếu độc lập. (R^2 = 1 - SSE/SST KHÔNG dùng được ở đây: với hai phía cùng chuẩn
    hóa, mức ngẫu nhiên của nó là -1 chứ không phải 0.)
    """
    sa, sb = np.linalg.norm(a_tr), np.linalg.norm(b_tr)
    if sa < EPS or sb < EPS:
        return float("nan")
    a_tr, a_te = a_tr / sa, a_te / sa
    b_tr, b_te = b_tr / sb, b_te / sb
    U, _, Vt = np.linalg.svd(a_tr.T @ b_tr, full_matrices=False)
    R = U @ Vt
    pred = a_te @ R
    den = np.linalg.norm(pred) * np.linalg.norm(b_te)
    if den < EPS:
        return float("nan")
    return float((pred * b_te).sum() / den)


def procrustes_cv(ZA, ZB, d_pca: Optional[int] = 64, seed: int = 0, *, cv: bool = True,
                  repeats: int = 1, dtype=np.float64) -> float:
    """Quay trực giao khớp trên nửa này, chấm điểm trên nửa kia; trung bình hai chiều (và `repeats` lần chia).

    Điểm = cosine Frobenius giữa ZA·R và ZB trên tập giữ ra ngoài: ~1 nếu ZB là bản quay của ZA, ~0 nếu độc lập.
    PCA (d_pca chiều, khớp trên tập khớp, tách riêng cho A và B) chạy trước để R không quá lớn so với n.
    d_pca=None: không PCA (dùng khi d nhỏ).
    cv=False: khớp và chấm trên cùng dữ liệu. Dùng để minh họa quá khớp khi n < d; nó KHÔNG phải số đo hợp lệ.
    """
    A, B = _as2d(ZA, dtype), _as2d(ZB, dtype)
    if A.shape != B.shape:
        raise ValueError("ZA và ZB phải cùng kích thước")
    n, d = A.shape
    if n < 8:
        raise ValueError("cần n >= 8")

    def reduce(tr_a, tr_b, te_a, te_b):
        if d_pca is None:
            mua, mub = tr_a.mean(0, keepdims=True), tr_b.mean(0, keepdims=True)
            return tr_a - mua, tr_b - mub, te_a - mua, te_b - mub
        p = int(min(d_pca, tr_a.shape[0] - 1, d))
        mua, Pa = _pca_fit(tr_a, p)
        mub, Pb = _pca_fit(tr_b, p)
        return (tr_a - mua) @ Pa, (tr_b - mub) @ Pb, (te_a - mua) @ Pa, (te_b - mub) @ Pb

    if not cv:
        a_tr, b_tr, a_te, b_te = reduce(A, B, A, B)
        return _procrustes_fit_score(a_tr, b_tr, a_te, b_te)

    scores = []
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        perm = rng.permutation(n)
        h = n // 2
        folds = [(perm[:h], perm[h:]), (perm[h:], perm[:h])]
        for tr, te in folds:
            a_tr, b_tr, a_te, b_te = reduce(A[tr], B[tr], A[te], B[te])
            scores.append(_procrustes_fit_score(a_tr, b_tr, a_te, b_te))
    return float(np.nanmean(scores))


# ------------------------------------------------------------------ truy hồi CSLS
def _unit(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), EPS)


def retrieval_csls(ZA, ZB, k: int = 10, *, center: bool = True, standardize: bool = False,
                   scale: Optional[np.ndarray] = None, return_ranks: bool = False, dtype=np.float64) -> Dict:
    """Tìm câu tương đương bằng CSLS(a, b) = 2 cos(a, b) - r_B(a) - r_A(b) (Conneau et al. 2018).

    r_B(a): trung bình cosine của a tới k láng giềng gần nhất trong B (giảm hubness).
    Trả về dict: acc1_ab, acc1_ba, acc1 (trung bình), mrr_ab, mrr_ba, mrr; nếu return_ranks=True còn
    `rank_ab`, `rank_ba` (hạng 1-indexed của đáp án đúng cho từng mục), dùng cho MRR theo từ ở Study 4.
    """
    A, B = prepare_pair(ZA, ZB, center=center, standardize=standardize, scale=scale, dtype=dtype)
    n = A.shape[0]
    k = int(min(k, n - 1))
    A, B = _unit(A), _unit(B)
    S = A @ B.T                                   # cosine, S[i, j] = a_i . b_j
    # r_B(a_i): top-k theo hàng; r_A(b_j): top-k theo cột
    rB = np.sort(S, axis=1)[:, -k:].mean(1)
    rA = np.sort(S, axis=0)[-k:, :].mean(0)
    C = 2.0 * S - rB[:, None] - rA[None, :]
    diag = np.diag(C)
    rank_ab = 1 + (C > diag[:, None]).sum(1)      # đếm đối thủ có điểm cao hơn đáp án đúng
    rank_ba = 1 + (C > diag[None, :]).sum(0)
    out = {
        "acc1_ab": float((rank_ab == 1).mean()), "acc1_ba": float((rank_ba == 1).mean()),
        "mrr_ab": float((1.0 / rank_ab).mean()), "mrr_ba": float((1.0 / rank_ba).mean()),
    }
    out["acc1"] = 0.5 * (out["acc1_ab"] + out["acc1_ba"])
    out["mrr"] = 0.5 * (out["mrr_ab"] + out["mrr_ba"])
    if return_ranks:
        out["rank_ab"], out["rank_ba"] = rank_ab, rank_ba
    return out