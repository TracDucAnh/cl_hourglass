"""U_k: thông tin dùng được (usable information, V-information; Xu et al. 2020) của danh tính ngôn ngữ Λ
trong biểu diễn Z_k sau khi trừ trung bình theo khái niệm.

Trừ trung bình theo khái niệm (mỗi nội dung c: Z_{c,l} - mean_l Z_{c,l}) loại phần mang nội dung C,
chỉ còn phần phụ thuộc ngôn ngữ; U_k là thay thế đo được cho Λ^res_k = I(Z_k; Λ | C) của mục 6.2.

    U = H(Λ) - H_V(Λ | Z')           (nat)
với H_V là cross-entropy giữ ra ngoài của một họ probe V cố định (tuyến tính hoặc MLP). Kiểm chéo
theo nhóm nội dung (GroupKFold) để cùng một nội dung không nằm ở cả tập khớp lẫn tập thử.
Cận lý thuyết: ở tầng embedding U / log L ≈ 1 (kiểm ở cổng G2 của Study 2a).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np


def center_by_group(Z: np.ndarray, groups: Sequence) -> np.ndarray:
    """Z - (trung bình các hàng cùng nhóm). Nhóm chỉ có một hàng cho ra hàng 0."""
    Z = np.asarray(Z, dtype=np.float64)
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    sums = np.zeros((inv.max() + 1, Z.shape[1]))
    np.add.at(sums, inv, Z)
    counts = np.bincount(inv).astype(np.float64)
    return Z - (sums / counts[:, None])[inv]


def _entropy_nats(y: np.ndarray) -> float:
    _, c = np.unique(y, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def make_probe(kind: Union[str, Callable], d_pca: Optional[int], seed: int, C: float = 1.0):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if callable(kind):
        return kind()
    steps = [StandardScaler()]
    if d_pca:
        steps.append(PCA(n_components=d_pca, random_state=seed))
    if kind == "linear":
        steps.append(LogisticRegression(C=C, max_iter=1000))
    elif kind == "mlp":
        steps.append(MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, early_stopping=True,
                                   max_iter=300, random_state=seed))
    else:
        raise ValueError(f"probe phải là 'linear', 'mlp' hoặc một hàm tạo estimator, nhận {kind!r}")
    return make_pipeline(*steps)


def usable_info(
    Z,
    lang: Sequence,
    groups: Sequence,
    probe: Union[str, Callable] = "linear",
    *,
    d_pca: Optional[int] = 64,
    n_splits: int = 5,
    seed: int = 0,
    center: bool = True,
    return_all: bool = False,
):
    """U = H(Λ) - H_V(Λ | Z') tính bằng nat.

    Z: [n, d] (mỗi hàng một (nội dung, ngôn ngữ)); lang: nhãn ngôn ngữ mỗi hàng; groups: id nội dung mỗi hàng.
    center=True: trừ trung bình theo `groups` (khuyến nghị; False chỉ để đối chứng).
    return_all=True: dict {U, U_bits, H_lang, ce, U_over_logL, U_over_H}.
    """
    from sklearn.model_selection import GroupKFold, cross_val_predict

    Z = np.asarray(Z, dtype=np.float64)
    lang = np.asarray(lang)
    groups = np.asarray(groups)
    if not (len(Z) == len(lang) == len(groups)):
        raise ValueError("Z, lang, groups phải cùng số hàng")
    classes, y = np.unique(lang, return_inverse=True)
    if len(classes) < 2:
        raise ValueError("cần ít nhất 2 ngôn ngữ")
    Zp = center_by_group(Z, groups) if center else Z
    n_groups = len(np.unique(groups))
    n_splits = int(min(n_splits, n_groups))
    n_train_min = len(Z) - int(np.ceil(len(Z) / n_splits)) - 1
    p = None if not d_pca else int(min(d_pca, Z.shape[1], max(n_train_min, 1)))
    est = make_probe(probe, p, seed)
    proba = cross_val_predict(est, Zp, y, groups=groups, cv=GroupKFold(n_splits=n_splits), method="predict_proba")
    ce = float(-np.log(np.clip(proba[np.arange(len(y)), y], 1e-12, 1.0)).mean())
    h = _entropy_nats(y)
    u = h - ce
    if not return_all:
        return float(u)
    return {
        "U": float(u), "U_bits": float(u / np.log(2)), "H_lang": h, "ce": ce,
        "U_over_logL": float(u / np.log(len(classes))), "U_over_H": float(u / h) if h > 0 else float("nan"),
    }