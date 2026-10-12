"""Lọc rò rỉ 13-gram giữa văn bản huấn luyện và bộ đánh giá (FLORES / NTREX / XNLI / UD).

Cách làm
  * "Đơn vị" = từ (chuỗi chữ/số/dấu kết hợp); với chữ Hán, Kana, Thái mỗi ký tự là một đơn vị.
  * Mỗi đơn vị -> băm 64-bit (blake2b, tất định) ; n-gram -> băm đa thức cuộn 64-bit (numpy, wrap-around).
  * Chỉ mục = mảng uint64 đã sắp xếp; tra bằng searchsorted (vector hóa).
  * Văn bản huấn luyện bị loại (mức tài liệu) nếu chứa >= 1 n-gram trùng bộ đánh giá.
  * Câu đánh giá ngắn hơn n đơn vị không sinh n-gram nên không phát hiện được (giới hạn đã biết).

CLI
    python scripts/leakage_check.py build  [--sources flores ntrex xnli ud belebele] [--n 13]
    python scripts/leakage_check.py check  FILE.jsonl [--field text]
    python scripts/leakage_check.py filter IN.jsonl OUT.jsonl [--field text]
"""
from __future__ import annotations

import argparse
import json
from hashlib import blake2b
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

import numpy as np
import regex

from clean import normalize_sentence
from common import (LEAK_INDEX_PATH, PARALLEL, TASKS, config_hash, load_config, log, read_jsonl,
                    record_processed, setup_logging, write_jsonl)

_CJK = "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u0e00-\u0e7f"
UNIT_RE = regex.compile(rf"[{_CJK}]|(?:(?![{_CJK}])[\p{{L}}\p{{N}}\p{{M}}])+")

_MASK = (1 << 64) - 1
_MULT = 0x9E3779B97F4A7C15  # số lẻ cố định
_cache: Dict[str, int] = {}


def _unit_hash(u: str) -> int:
    h = _cache.get(u)
    if h is None:
        if len(_cache) > 2_000_000:
            _cache.clear()
        h = int.from_bytes(blake2b(u.encode("utf-8"), digest_size=8).digest(), "little")
        _cache[u] = h
    return h


def ngram_keys(text: str, n: int = 13) -> np.ndarray:
    units = UNIT_RE.findall(text.lower())
    if len(units) < n:
        return np.empty(0, dtype=np.uint64)
    h = np.fromiter((_unit_hash(u) for u in units), dtype=np.uint64, count=len(units))
    m = len(h) - n + 1
    keys = np.zeros(m, dtype=np.uint64)
    for j in range(n):
        keys += h[j : j + m] * np.uint64(pow(_MULT, j + 1, 1 << 64))  # wrap-around uint64
    return keys


class LeakageIndex:
    def __init__(self, keys: np.ndarray, n: int = 13, meta: dict | None = None):
        self.keys, self.n, self.meta = keys, n, meta or {}

    @classmethod
    def build(cls, texts: Iterable[str], n: int = 13, meta: dict | None = None) -> "LeakageIndex":
        chunks: List[np.ndarray] = []
        for t in texts:
            k = ngram_keys(t, n)
            if k.size:
                chunks.append(k)
        keys = np.unique(np.concatenate(chunks)) if chunks else np.empty(0, dtype=np.uint64)
        return cls(keys, n, meta)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(tmp, keys=self.keys, n=np.int64(self.n), meta=np.array(json.dumps(self.meta)))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "LeakageIndex":
        z = np.load(path, allow_pickle=False)
        return cls(z["keys"], int(z["n"]), json.loads(str(z["meta"])))

    def hits(self, text: str) -> int:
        k = ngram_keys(text, self.n)
        if k.size == 0 or self.keys.size == 0:
            return 0
        pos = np.searchsorted(self.keys, k)
        pos[pos == self.keys.size] = self.keys.size - 1
        return int((self.keys[pos] == k).sum())

    def is_leaky(self, text: str) -> bool:
        return self.hits(text) > 0


# ---------------------------------------------------------------- nguồn đánh giá
def _flores(cfg) -> Iterator[str]:
    for name in ("flores_dev.jsonl", "flores_devtest.jsonl"):
        p = PARALLEL / name
        if p.exists():
            for r in read_jsonl(p):
                yield r["text"]
        else:
            log.warning("thiếu %s (chạy build_parallel.py trước)", p)


def _ntrex(cfg) -> Iterator[str]:
    p = PARALLEL / "ntrex.jsonl"
    if p.exists():
        for r in read_jsonl(p):
            yield r["text"]
    else:
        log.warning("thiếu %s", p)


def _xnli(cfg) -> Iterator[str]:
    d = TASKS / "xnli"
    files = sorted(d.glob("*.jsonl")) if d.exists() else []
    if not files:
        log.warning("thiếu %s (chạy build_tasks.py xnli)", d)
    for p in files:
        for r in read_jsonl(p):
            yield r["premise"]
            yield r["hypothesis"]


def _ud(cfg) -> Iterator[str]:
    d = TASKS / "upos"
    files = sorted(d.glob("*.jsonl")) if d.exists() else []
    if not files:
        log.warning("thiếu %s (chạy build_tasks.py upos)", d)
    for p in files:
        for r in read_jsonl(p):
            yield r["text"]


def _belebele(cfg) -> Iterator[str]:
    d = TASKS / "belebele"
    for p in sorted(d.glob("*.jsonl")) if d.exists() else []:
        for r in read_jsonl(p):
            yield r["passage"]
            yield r["question"]
            yield from r["answers"]


SOURCES = {"flores": _flores, "ntrex": _ntrex, "xnli": _xnli, "ud": _ud, "belebele": _belebele}


def build_index(sources: List[str], n: int, path: Path = LEAK_INDEX_PATH) -> LeakageIndex:
    cfg = load_config()

    def texts():
        for s in sources:
            cnt = 0
            for t in SOURCES[s](cfg):
                cnt += 1
                yield normalize_sentence(t)
            log.info("nguồn %-9s: %d văn bản", s, cnt)

    idx = LeakageIndex.build(texts(), n, meta={"sources": sources, "n": n})
    idx.save(path)
    log.info("chỉ mục: %d n-gram duy nhất -> %s", idx.keys.size, path)
    record_processed("leakage_index", "leakage_check.py", sources, [path], config_hash({"sources": sources, "n": n}),
                     n_ngrams=int(idx.keys.size), ngram=n)
    return idx


def load_or_none(path: Path = LEAK_INDEX_PATH) -> LeakageIndex | None:
    return LeakageIndex.load(path) if Path(path).exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--sources", nargs="+", default=["flores", "ntrex", "xnli", "ud"], choices=list(SOURCES))
    b.add_argument("--n", type=int, default=None)
    b.add_argument("--out", default=str(LEAK_INDEX_PATH))
    for name in ("check", "filter"):
        s = sub.add_parser(name)
        s.add_argument("inp")
        if name == "filter":
            s.add_argument("out")
        s.add_argument("--field", default="text")
        s.add_argument("--index", default=str(LEAK_INDEX_PATH))
    args = ap.parse_args()
    setup_logging()

    if args.cmd == "build":
        n = args.n or int(load_config()["mono"]["leakage_ngram"])
        build_index(args.sources, n, Path(args.out))
        return

    idx = load_or_none(Path(args.index))
    if idx is None:
        raise SystemExit(f"Chưa có chỉ mục {args.index}; chạy: leakage_check.py build")
    if args.cmd == "check":
        tot = leaky = 0
        for r in read_jsonl(args.inp):
            tot += 1
            leaky += idx.is_leaky(normalize_sentence(r[args.field]))
        log.info("%d/%d (%.3f%%) bản ghi có rò rỉ %d-gram", leaky, tot, 100 * leaky / max(tot, 1), idx.n)
    else:
        kept = [0, 0]

        def gen():
            for r in read_jsonl(args.inp):
                kept[1] += 1
                if idx.is_leaky(normalize_sentence(r[args.field])):
                    continue
                kept[0] += 1
                yield r

        write_jsonl(args.out, gen())
        log.info("giữ %d/%d bản ghi", kept[0], kept[1])


if __name__ == "__main__":
    main()
