"""Mô hình hỗn hợp tuyến tính (REML) với hiệu ứng ngẫu nhiên theo cặp ngôn ngữ và/hoặc mô hình.

Bọc statsmodels.MixedLM. Kế hoạch yêu cầu "mô hình hỗn hợp với hiệu ứng ngẫu nhiên theo cặp ngôn ngữ và mô hình"
(P1, P3, S5): đó là hai yếu tố ngẫu nhiên CHÉO nhau, nên dùng `crossed=['pair', 'model']`; statsmodels biểu diễn
hiệu ứng chéo bằng một nhóm duy nhất và các thành phần phương sai (vc_formula).

Chưa có: mô hình hỗn hợp logistic (S4 "Logistic hỗn hợp"). statsmodels chỉ có BinomialBayesMixedGLM (xấp xỉ biến phân);
nếu cần, dùng nó hoặc glmmTMB/lme4 bên R. Không giả vờ hỗ trợ ở đây.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class MixedResult:
    fixed: pd.DataFrame                     # coef, se, z, p, ci_low, ci_high (index = tên hệ số)
    random: Dict[str, float]                # thành phần phương sai
    scale: float                            # phương sai dư
    converged: bool
    llf: float
    n_obs: int
    reml: bool
    warnings: list = field(default_factory=list)
    result: object = field(default=None, repr=False)

    def coef(self, name: str) -> float:
        return float(self.fixed.loc[name, "coef"])


def mixed_model(
    df: pd.DataFrame,
    formula: str,
    *,
    group: Optional[str] = None,
    crossed: Optional[Sequence[str]] = None,
    re_formula: Optional[str] = None,
    reml: bool = True,
    method: Sequence[str] = ("lbfgs", "powell", "nm"),
    maxiter: int = 500,
) -> MixedResult:
    """Khớp `formula` (cú pháp patsy, ví dụ 'G ~ log_B + log_L') với hiệu ứng ngẫu nhiên.

    group: một yếu tố nhóm (chặn ngẫu nhiên; re_formula='~x' để thêm độ dốc ngẫu nhiên).
    crossed: danh sách cột là các yếu tố ngẫu nhiên chéo (mỗi cột một chặn ngẫu nhiên). Không dùng cùng `group`.
    Thử lần lượt các bộ tối ưu trong `method` cho đến khi hội tụ.
    """
    import statsmodels.formula.api as smf

    if (group is None) == (crossed is None):
        raise ValueError("chọn đúng một trong `group` hoặc `crossed`")
    d = df.reset_index(drop=True).copy()
    kwargs = {}
    if crossed is not None:
        d["__all__"] = 1
        groups = d["__all__"]
        kwargs["vc_formula"] = {c: f"0 + C({c})" for c in crossed}
        kwargs["re_formula"] = "0"
    else:
        groups = d[group]
        if re_formula is not None:
            kwargs["re_formula"] = re_formula

    model = smf.mixedlm(formula, d, groups=groups, **kwargs)
    res, caught = None, []
    for m in method:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                r = model.fit(reml=reml, method=m, maxiter=maxiter)
            except Exception as e:  # noqa: BLE001
                caught.append(f"{m}: {type(e).__name__}: {e}")
                continue
        caught += [f"{m}: {str(x.message)[:120]}" for x in w]
        res = r
        if getattr(r, "converged", False):
            break
    if res is None:
        raise RuntimeError("mô hình hỗn hợp không khớp được với mọi bộ tối ưu: " + " | ".join(caught))

    k_fe = len(res.fe_params)
    fe = pd.DataFrame({
        "coef": res.fe_params.values,
        "se": res.bse_fe.values,
        "z": (res.fe_params / res.bse_fe).values,
        "p": res.pvalues[:k_fe].values,
    }, index=res.fe_params.index)
    ci = res.conf_int().iloc[:k_fe]
    fe["ci_low"], fe["ci_high"] = ci.iloc[:, 0].values, ci.iloc[:, 1].values

    rnd: Dict[str, float] = {}
    if crossed is not None:
        for name, v in zip(model.exog_vc.names, np.atleast_1d(res.vcomp)):
            rnd[str(name)] = float(v)
    else:
        cov = np.atleast_2d(np.asarray(res.cov_re))
        for i, nm in enumerate(res.model.exog_re_names if hasattr(res.model, "exog_re_names") else range(len(cov))):
            rnd[f"{group}:{nm}"] = float(cov[i, i])
    return MixedResult(fe, rnd, float(res.scale), bool(getattr(res, "converged", False)), float(res.llf),
                       int(res.nobs), reml, caught, res)