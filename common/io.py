"""I/O dùng chung cho mọi study.

Gồm:
  * đường dẫn tới data/ (tôn trọng HOURGLASS_DATA_DIR, giống data/scripts/common.py);
  * đọc dữ liệu song song {id, lang, text} -> ParallelData (cùng thứ tự id ở mọi ngôn ngữ);
  * set_seed, config_hash (cùng công thức với data/scripts/common.py), git_commit;
  * bảng kết quả parquet với schema chung: mỗi bảng đều mang config_hash, seed, git_commit,
    created_at, để mỗi số trong bài truy ngược được về một lần chạy (quy tắc ràng buộc của kế hoạch).

Không import torch ở đây (set_seed chỉ gọi torch nếu đã được import sẵn).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

PROVENANCE_COLS = ("config_hash", "seed", "git_commit", "created_at")


# ------------------------------------------------------------------ đường dẫn
def data_dir() -> Path:
    """data/ của kho mã; đặt HOURGLASS_DATA_DIR để trỏ sang bản sao khác."""
    return Path(os.environ.get("HOURGLASS_DATA_DIR") or REPO_ROOT / "data").resolve()


def parallel_path(dataset: str | Path) -> Path:
    """'flores_devtest' -> data/processed/parallel/flores_devtest.jsonl; hoặc một đường dẫn jsonl."""
    p = Path(dataset)
    if p.suffix in (".jsonl", ".gz") and p.exists():
        return p
    q = data_dir() / "processed" / "parallel" / f"{dataset}.jsonl"
    if not q.exists():
        raise FileNotFoundError(
            f"Không thấy {q}. Chạy data/scripts/build_parallel.py (flores/ntrex) trước."
        )
    return q


def _read_jsonl(path: Path) -> Iterable[dict]:
    import gzip

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ------------------------------------------------------------------ dữ liệu song song
@dataclass
class ParallelData:
    """texts[lang][i] là câu của ids[i] ở ngôn ngữ lang; cùng id = cùng nội dung C."""

    dataset: str
    ids: List[str]
    langs: List[str]
    texts: Dict[str, List[str]]

    def __len__(self) -> int:
        return len(self.ids)


def load_parallel(
    dataset: str | Path,
    langs: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> ParallelData:
    """Đọc jsonl {id, lang, text} thành các cột song song.

    langs: thứ tự ngôn ngữ mong muốn (mặc định: thứ tự xuất hiện trong tệp).
    limit: chỉ lấy `limit` id đầu tiên (để chạy thử).
    Mọi ngôn ngữ phải có đúng cùng tập id (build_parallel.py đảm bảo bằng phép giao).
    """
    path = parallel_path(dataset)
    by_lang: Dict[str, Dict[str, str]] = {}
    order: List[str] = []
    seen_ids = set()
    for r in _read_jsonl(path):
        l = r["lang"]
        if langs is not None and l not in langs:
            continue
        by_lang.setdefault(l, {})[r["id"]] = r["text"]
        if r["id"] not in seen_ids:
            seen_ids.add(r["id"])
            order.append(r["id"])
    use = list(langs) if langs is not None else list(by_lang)
    missing = [l for l in use if l not in by_lang]
    if missing:
        raise KeyError(f"{path.name}: thiếu ngôn ngữ {missing}; có {sorted(by_lang)}")
    id_sets = {l: set(by_lang[l]) for l in use}
    common = set.intersection(*id_sets.values())
    for l in use:
        if id_sets[l] != common:
            raise ValueError(
                f"{path.name}: ngôn ngữ {l} có {len(id_sets[l])} id, giao của các ngôn ngữ là {len(common)}"
            )
    ids = [i for i in order if i in common]
    if limit is not None:
        ids = ids[: int(limit)]
    texts = {l: [by_lang[l][i] for i in ids] for l in use}
    return ParallelData(dataset=Path(str(dataset)).stem, ids=ids, langs=use, texts=texts)


# ------------------------------------------------------------------ seed, hash, git
def set_seed(seed: int, deterministic: bool = False) -> int:
    """Đặt seed cho random, numpy và (nếu đã import) torch."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return seed


def config_hash(obj: Any) -> str:
    """Cùng công thức với data/scripts/common.py để hash khớp giữa hai nơi."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def git_commit(cwd: Optional[Path] = None) -> str:
    """Commit hiện tại của kho mã, thêm '-dirty' nếu có thay đổi chưa commit; 'nogit' nếu không có git."""
    cwd = Path(cwd or REPO_ROOT)
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=10
        )
        if sha.returncode != 0:
            return "nogit"
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd, capture_output=True, text=True, timeout=10,
        )
        return sha.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return "nogit"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_provenance(config: Any = None, seed: Optional[int] = None) -> Dict[str, Any]:
    return {
        "config_hash": config_hash(config if config is not None else {}),
        "seed": -1 if seed is None else int(seed),
        "git_commit": git_commit(),
        "created_at": utcnow(),
    }


# ------------------------------------------------------------------ schema bảng kết quả
class SchemaError(ValueError):
    pass


# cột bắt buộc theo loại bảng; cột thêm được phép
SCHEMAS: Dict[str, Dict[str, type]] = {
    # Gk, Dk, CKA, ... theo (cặp ngôn ngữ, tầng): results/*/curves.parquet
    "curves": {
        "family": str, "model": str, "dataset": str, "lang_a": str, "lang_b": str,
        "layer": int, "rho": float, "metric": str, "value": float,
    },
    # đặc trưng đường cong (k*, rho*, w, S): results/*/features.parquet
    "features": {
        "family": str, "model": str, "dataset": str, "lang_a": str, "lang_b": str,
        "metric": str, "k_star": float, "rho_star": float, "width": float, "width_rel": float,
        "amplitude": float,
    },
    "generic": {},
}
for _s in SCHEMAS.values():
    for _c, _t in (("config_hash", str), ("seed", int), ("git_commit", str), ("created_at", str)):
        _s[_c] = _t


def validate_table(df: pd.DataFrame, kind: str) -> None:
    if kind not in SCHEMAS:
        raise SchemaError(f"kind không hợp lệ: {kind!r}; chọn trong {sorted(SCHEMAS)}")
    missing = [c for c in SCHEMAS[kind] if c not in df.columns]
    if missing:
        raise SchemaError(f"bảng '{kind}' thiếu cột bắt buộc: {missing}")
    for c, t in SCHEMAS[kind].items():
        s = df[c]
        if t is str:
            ok = pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s)
        elif t is int:
            ok = pd.api.types.is_integer_dtype(s)
        else:
            ok = pd.api.types.is_numeric_dtype(s)
        if not ok:
            raise SchemaError(f"cột {c!r} của bảng '{kind}' phải kiểu {t.__name__}, đang là {s.dtype}")


def write_table(
    path: str | Path,
    df: pd.DataFrame,
    *,
    kind: str = "generic",
    config: Any = None,
    seed: Optional[int] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Ghi parquet (nguyên tử) kèm cột nguồn gốc và siêu dữ liệu.

    Các cột config_hash/seed/git_commit/created_at được thêm nếu chưa có.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prov = run_provenance(config, seed)
    out = df.copy()
    for c in PROVENANCE_COLS:
        if c not in out.columns:
            out[c] = prov[c]
    out["seed"] = out["seed"].astype("int64")
    validate_table(out, kind)

    table = pa.Table.from_pandas(out, preserve_index=False)
    meta = {**prov, "kind": kind, **(extra_meta or {})}
    md = dict(table.schema.metadata or {})
    md[b"hourglass"] = json.dumps(meta, default=str, ensure_ascii=False).encode("utf-8")
    table = table.replace_schema_metadata(md)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)
    return path


def read_table(path: str | Path, kind: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if kind is not None:
        validate_table(df, kind)
    return df


def read_meta(path: str | Path) -> Dict[str, Any]:
    import pyarrow.parquet as pq

    md = pq.read_schema(path).metadata or {}
    return json.loads(md.get(b"hourglass", b"{}").decode("utf-8"))