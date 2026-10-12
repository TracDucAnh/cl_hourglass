"""Tiện ích dùng chung cho mọi script trong data/scripts.

Gồm: đường dẫn, cấu hình, đọc/ghi jsonl, manifest.json (có khóa tệp),
tải tệp qua HTTP, kế hoạch ngân sách byte (byte premium).

Đặt biến môi trường HOURGLASS_DATA_DIR để chạy trên một bản sao khác của data/
(selftest.py dùng cách này).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

DATA_DIR = Path(os.environ.get("HOURGLASS_DATA_DIR") or Path(__file__).resolve().parent.parent).resolve()
RAW = DATA_DIR / "raw"
PROC = DATA_DIR / "processed"
PARALLEL = PROC / "parallel"
MONO = PROC / "mono"
TASKS = PROC / "tasks"
TYPOLOGY = PROC / "typology"
MANIFEST_PATH = DATA_DIR / "manifest.json"
CONFIG_PATH = DATA_DIR / "config.yaml"
PREMIUM_PATH = MONO / "byte_premium.json"
LEAK_INDEX_PATH = PROC / "leakage_index.npz"

log = logging.getLogger("hourglass")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- config
_CFG: Optional[dict] = None


def load_config(path: Optional[Path] = None) -> dict:
    import yaml

    global _CFG
    if _CFG is None or path is not None:
        with open(path or CONFIG_PATH, encoding="utf-8") as f:
            _CFG = yaml.safe_load(f)
    return _CFG


def config_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def flores_code(cfg: dict, lang: str) -> str:
    return cfg["languages"][lang]["flores"]


# ---------------------------------------------------------------- io
def open_text(path: Path | str, mode: str = "rt"):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8", newline="\n" if "w" in mode else None)


def read_jsonl(path: Path | str) -> Iterator[dict]:
    with open_text(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def atomic_write_text(path: Path | str, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def write_jsonl(path: Path | str, rows: Iterable[dict]) -> int:
    """Ghi nguyên tử; trả về số dòng."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    n = 0
    if path.suffix == ".gz":
        f = gzip.open(tmp, "wt", encoding="utf-8")
    else:
        f = open(tmp, "w", encoding="utf-8", newline="\n")
    with f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


class ShardWriter:
    """Ghi jsonl theo shard, quay sang shard mới khi vượt max_bytes (byte chưa nén)."""

    def __init__(self, outdir: Path, prefix: str, max_bytes: int, gz: bool = False):
        self.outdir, self.prefix, self.max_bytes, self.gz = Path(outdir), prefix, int(max_bytes), gz
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.idx = -1
        self.f = None
        self.cur = 0
        self.total = 0
        self.files: List[Path] = []

    def _open_next(self):
        if self.f:
            self.f.close()
        self.idx += 1
        ext = ".jsonl.gz" if self.gz else ".jsonl"
        p = self.outdir / f"{self.prefix}_{self.idx:05d}{ext}"
        self.f = gzip.open(p, "wt", encoding="utf-8") if self.gz else open(p, "w", encoding="utf-8", newline="\n")
        self.files.append(p)
        self.cur = 0

    def write(self, row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False) + "\n"
        n = len(line.encode("utf-8"))
        if self.f is None or (self.cur > 0 and self.cur + n > self.max_bytes):
            self._open_next()
        self.f.write(line)
        self.cur += n
        self.total += n

    def close(self) -> None:
        if self.f:
            self.f.close()
            self.f = None


# ---------------------------------------------------------------- checksum / manifest
def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_info(path: Path | str) -> dict:
    p = Path(path)
    return {"bytes": p.stat().st_size, "sha256": sha256_file(p)}


def dir_files_info(root: Path | str) -> Dict[str, dict]:
    root = Path(root)
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.name.startswith(".") and p.suffix not in (".part", ".tmp", ".lock"):
            out[p.relative_to(root).as_posix()] = file_info(p)
    return out


@contextmanager
def _file_lock(path: Path, timeout: float = 60.0):
    lock = Path(str(path) + ".lock")
    t0 = time.time()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if time.time() - t0 > timeout:  # khóa cũ bị bỏ lại
                try:
                    os.remove(lock)
                except OSError:
                    pass
                t0 = time.time()
            time.sleep(0.1)
    try:
        yield
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"schema_version": 1, "raw": {}, "processed": {}, "verification": {}}


def update_manifest(section: str, key: str, entry: dict) -> None:
    """section: 'raw' | 'processed' | 'verification'."""
    with _file_lock(MANIFEST_PATH):
        m = load_manifest()
        m.setdefault(section, {})[key] = entry
        m["updated"] = utcnow()
        atomic_write_text(MANIFEST_PATH, json.dumps(m, ensure_ascii=False, indent=2, sort_keys=False) + "\n")


def record_raw(name: str, root: Path, source: Any, revision: Optional[str] = None, **extra) -> dict:
    entry = {
        "source": source,
        "revision": revision,
        "fetched_at": utcnow(),
        "files": dir_files_info(root),
        **extra,
    }
    update_manifest("raw", name, entry)
    return entry


def record_processed(key: str, script: str, inputs: List[str], paths: List[Path], cfg_hash: str, **extra) -> dict:
    entry = {
        "script": script,
        "inputs": inputs,
        "built_at": utcnow(),
        "config_hash": cfg_hash,
        "files": {Path(p).relative_to(DATA_DIR).as_posix(): file_info(p) for p in paths},
        **extra,
    }
    update_manifest("processed", key, entry)
    return entry


# ---------------------------------------------------------------- http
def http_session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(total=5, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504])))
    s.headers["User-Agent"] = "hourglass-data/1.0"
    return s


def download(url: str, dest: Path | str, *, force: bool = False, timeout: int = 60) -> Path:
    dest = Path(dest)
    if dest.exists() and not force:
        log.info("đã có, bỏ qua: %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    s = http_session()
    log.info("tải %s", url)
    with s.get(url, stream=True, timeout=timeout, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        try:
            from tqdm import tqdm

            bar = tqdm(total=total or None, unit="B", unit_scale=True, desc=dest.name, leave=False)
        except ImportError:
            bar = None
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                if bar is not None:
                    bar.update(len(chunk))
        if bar is not None:
            bar.close()
    os.replace(tmp, dest)
    return dest


def url_ok(url: str, timeout: int = 30) -> tuple[bool, str]:
    """Kiểm tra URL còn sống mà không tải nội dung."""
    try:
        s = http_session()
        r = s.get(url, stream=True, timeout=timeout, allow_redirects=True)
        code = r.status_code
        r.close()
        return code < 400, f"HTTP {code}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------- byte premium / ngân sách
def load_premium(path: Path = PREMIUM_PATH) -> Dict[str, float]:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Chưa có {path}. Chạy trước: build_parallel.py (flores) rồi byte_premium.py."
        )
    with open(path, encoding="utf-8") as f:
        return {k: v["premium"] for k, v in json.load(f)["languages"].items()}


def byte_budgets(prem: Dict[str, float], cfg: dict, scale: float = 1.0) -> Dict[str, Dict[str, int]]:
    """Ngân sách byte mỗi (cấu hình L, ngôn ngữ) = (T/L) * bytes_per_token_en * prem_l * scale."""
    m = cfg["mono"]
    out: Dict[str, Dict[str, int]] = {}
    for name, langs in cfg["configs"].items():
        base = float(m["T_tokens"]) / len(langs) * float(m["bytes_per_token_en"]) * scale
        out[name] = {l: int(round(base * prem[l])) for l in langs}
    return out


def per_language_budget(prem: Dict[str, float], cfg: dict, scale: float = 1.0) -> Dict[str, int]:
    """Số byte tối đa cần giữ cho từng ngôn ngữ (lấy max qua các cấu hình lồng nhau);
    ngôn ngữ chỉ-đếm-tần-suất nhận extra_bytes."""
    res: Dict[str, int] = {}
    for d in byte_budgets(prem, cfg, scale).values():
        for l, b in d.items():
            res[l] = max(res.get(l, 0), b)
    extra = int(float(cfg["mono"]["extra_bytes"]) * scale)
    for l in cfg["wordfreq_languages"]:
        res.setdefault(l, extra)
    return res


def mono_raw_dir(lang: str) -> Path:
    return RAW / ("fineweb" if lang == "eng" else "fineweb2") / lang
