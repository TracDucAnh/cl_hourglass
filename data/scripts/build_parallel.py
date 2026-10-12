"""raw/ -> processed/parallel/*.jsonl   (định dạng mỗi dòng: {id, lang, text}).

    python scripts/build_parallel.py                 # flores + ntrex
    python scripts/build_parallel.py flores
    python scripts/build_parallel.py opus100         # cần raw/opus100 và (khuyến nghị) leakage_index.npz

Một 'id' chung cho cùng một nội dung ở mọi ngôn ngữ (đơn vị bootstrap theo cụm nội dung).
FLORES/NTREX: lấy giao các mục song song của các ngôn ngữ; NFKC.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from clean import ExactDeduper, normalize_sentence
from common import (PARALLEL, RAW, config_hash, flores_code, load_config, log, record_processed, setup_logging,
                    write_jsonl)


def _read_lines(p: Path) -> List[str]:
    with open(p, encoding="utf-8") as f:
        return [normalize_sentence(l.rstrip("\n")) for l in f]


def _intersect_rows(prefix: str, texts: Dict[str, List[str]], langs: List[str]):
    """texts[lang] là các dòng căn theo vị trí. Giữ vị trí mà mọi ngôn ngữ đều khác rỗng."""
    sizes = {l: len(v) for l, v in texts.items()}
    if len(set(sizes.values())) != 1:
        raise RuntimeError(f"{prefix}: số dòng khác nhau giữa các ngôn ngữ: {sizes}")
    n = next(iter(sizes.values()))
    keep = [i for i in range(n) if all(texts[l][i] for l in langs)]
    if len(keep) < n:
        log.warning("%s: bỏ %d/%d mục có ngôn ngữ rỗng", prefix, n - len(keep), n)
    for i in keep:
        for l in langs:
            yield {"id": f"{prefix}_{i:04d}", "lang": l, "text": texts[l][i]}


def build_flores(cfg, langs: List[str]) -> None:
    for split in ("dev", "devtest"):
        texts = {}
        for l in langs:
            p = RAW / "flores" / split / f"{flores_code(cfg, l)}.{split}"
            if not p.exists():
                raise SystemExit(f"Thiếu {p}. Chạy: python scripts/fetch.py flores")
            texts[l] = _read_lines(p)
        out = PARALLEL / f"flores_{split}.jsonl"
        n = write_jsonl(out, _intersect_rows(f"flores_{split}", texts, langs))
        log.info("flores %-7s: %d dòng (%d ngôn ngữ) -> %s", split, n, len(langs), out.name)
        record_processed(f"parallel/flores_{split}", "build_parallel.py", ["raw/flores"], [out],
                         config_hash({"langs": langs, "split": split}), n_rows=n, languages=langs)


def build_ntrex(cfg, langs: List[str]) -> None:
    from fetch import ntrex_filename

    texts = {}
    for l in langs:
        p = RAW / "ntrex" / "NTREX-128" / ntrex_filename(cfg["languages"][l]["ntrex"])
        if not p.exists():
            raise SystemExit(f"Thiếu {p}. Chạy: python scripts/fetch.py ntrex")
        texts[l] = _read_lines(p)
    out = PARALLEL / "ntrex.jsonl"
    n = write_jsonl(out, _intersect_rows("ntrex", texts, langs))
    log.info("ntrex: %d dòng -> %s", n, out.name)
    record_processed("parallel/ntrex", "build_parallel.py", ["raw/ntrex"], [out], config_hash({"langs": langs}),
                     n_rows=n, languages=langs)


def build_opus(cfg, langs: List[str], max_pairs: int | None) -> None:
    import pandas as pd

    from fetch import opus_pair
    from leakage_check import load_or_none

    idx = load_or_none()
    if idx is None:
        log.warning("Chưa có leakage_index.npz: OPUS-100 sẽ KHÔNG được lọc rò rỉ. "
                    "Khuyến nghị chạy leakage_check.py build trước.")
    for l in langs:
        iso = cfg["languages"][l]["opus"]
        pair = opus_pair(cfg, l)
        files = sorted((RAW / "opus100" / pair).glob("train-*.parquet"))
        if not files:
            raise SystemExit(f"Thiếu raw/opus100/{pair}/train-*.parquet. Chạy: fetch.py opus100")
        ded = ExactDeduper()
        stats = {"in": 0, "empty": 0, "dup": 0, "leak": 0}

        def gen():
            k = 0
            for f in files:
                for tr in pd.read_parquet(f)["translation"]:
                    stats["in"] += 1
                    en, xx = normalize_sentence(tr.get("en", "")), normalize_sentence(tr.get(iso, ""))
                    if not en or not xx:
                        stats["empty"] += 1
                        continue
                    if not ded.add(en + "\t" + xx):
                        stats["dup"] += 1
                        continue
                    if idx is not None and (idx.is_leaky(en) or idx.is_leaky(xx)):
                        stats["leak"] += 1
                        continue
                    yield {"id": f"opus100_{pair}_{k:07d}", "lang": "eng", "text": en}
                    yield {"id": f"opus100_{pair}_{k:07d}", "lang": l, "text": xx}
                    k += 1
                    if max_pairs and k >= max_pairs:
                        return

        out = PARALLEL / f"opus100_{pair}.jsonl"
        n = write_jsonl(out, gen())
        log.info("opus100 %s: %d cặp (vào=%d rỗng=%d trùng=%d rò_rỉ=%d)", pair, n // 2, stats["in"], stats["empty"],
                 stats["dup"], stats["leak"])
        record_processed(f"parallel/opus100_{pair}", "build_parallel.py", [f"raw/opus100/{pair}"], [out],
                         config_hash({"pair": pair, "max": max_pairs}), n_pairs=n // 2, **{f"drop_{k}": v for k, v in stats.items()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", help="flores | ntrex | opus100 (mặc định: flores ntrex)")
    ap.add_argument("--langs", nargs="+")
    ap.add_argument("--max-pairs", type=int, default=None, help="giới hạn số cặp OPUS-100 mỗi ngôn ngữ")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config()
    targets = args.targets or ["flores", "ntrex"]
    bad = [t for t in targets if t not in ("flores", "ntrex", "opus100")]
    if bad:
        raise SystemExit(f"Mục tiêu không hợp lệ: {bad}")
    for t in targets:
        if t == "flores":
            build_flores(cfg, args.langs or cfg["study_languages"])
        elif t == "ntrex":
            build_ntrex(cfg, args.langs or cfg["study_languages"])
        else:
            build_opus(cfg, args.langs or cfg["opus_languages"], args.max_pairs)


if __name__ == "__main__":
    main()
