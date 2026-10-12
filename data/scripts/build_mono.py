"""raw/fineweb2 (+ raw/fineweb cho tiếng Anh) -> processed/mono/{lang}/shard_*.jsonl

Mỗi tài liệu: NFKC -> bỏ nếu quá ngắn -> lọc trùng chính xác -> lọc rò rỉ 13-gram
(FLORES/NTREX/XNLI/UD) -> ghi vào shard, dừng khi đủ ngân sách byte của ngôn ngữ:
    ngân sách = max qua các cấu hình lồng nhau của  (T/L) * bytes_per_token_en * byte_premium_l

    python scripts/build_mono.py                      # mọi ngôn ngữ có raw
    python scripts/build_mono.py --langs fra zho --scale 0.01
    python scripts/build_mono.py --no-leakage         # KHÔNG khuyến nghị (chỉ để thử)
Mỗi thư mục ngôn ngữ có _meta.json (thống kê, ngân sách từng cấu hình L) cho study3/build_corpus.py.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from clean import ExactDeduper, clean_document
from common import (DATA_DIR, MONO, ShardWriter, byte_budgets, config_hash, load_config, load_premium, log, mono_raw_dir,
                    per_language_budget, read_jsonl, record_processed, setup_logging, utcnow)
from leakage_check import load_or_none


def build_language(lang: str, budget: int, plan: dict, cfg: dict, idx, args) -> dict | None:
    src = mono_raw_dir(lang)
    parts = sorted(src.glob("part_*.jsonl.gz"))
    if not parts:
        log.error("%s: không có raw tại %s (chạy fetch.py fineweb2)", lang, src)
        return None
    outdir, tmp = MONO / lang, MONO / f".{lang}.tmp"
    if outdir.exists():
        if not args.force:
            log.info("%s: đã có, bỏ qua (dùng --force để làm lại)", lang)
            return None
        shutil.rmtree(outdir)
    if tmp.exists():
        shutil.rmtree(tmp)

    m = cfg["mono"]
    writer = ShardWriter(tmp, "shard", int(m["shard_bytes"]))
    ded, st = ExactDeduper(), Counter()
    kept_bytes = 0
    try:
        from tqdm import tqdm

        bar = tqdm(total=budget, unit="B", unit_scale=True, desc=lang)
    except ImportError:
        bar = None
    done = False
    for part in parts:
        for row in read_jsonl(part):
            st["raw_docs"] += 1
            txt = row.get("text") or ""
            st["raw_bytes"] += len(txt.encode("utf-8"))
            doc = clean_document(txt, int(m["min_chars"]))
            if doc is None:
                st["drop_short_or_bad"] += 1
                continue
            if not ded.add(doc):
                st["drop_duplicate"] += 1
                continue
            if idx is not None and idx.is_leaky(doc):
                st["drop_leak"] += 1
                continue
            b = len(doc.encode("utf-8"))
            writer.write({"id": f"{lang}-{st['kept_docs']:09d}", "lang": lang, "text": doc})
            st["kept_docs"] += 1
            kept_bytes += b
            if bar is not None:
                bar.update(b)
            if kept_bytes >= budget:
                done = True
                break
        if done:
            break
    writer.close()
    if bar is not None:
        bar.close()

    meta = {
        "lang": lang,
        "built_at": utcnow(),
        "purpose": "train" if any(lang in l for l in cfg["configs"].values()) else "wordfreq_only",
        "budget_bytes": budget,
        "budget_per_config_bytes": {k: v[lang] for k, v in plan.items() if lang in v},
        "kept_bytes": kept_bytes,
        "shortfall": kept_bytes < budget,
        "leakage_filter": idx is not None,
        "leakage_ngram": idx.n if idx is not None else None,
        "stats": dict(st),
        "n_shards": len(writer.files),
    }
    (tmp / "_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.replace(outdir)
    if meta["shortfall"]:
        log.warning("%s: THIẾU dữ liệu (%.1f/%.1f MB). Tải thêm raw (tăng --scale hoặc raw_margin).", lang,
                    kept_bytes / 1e6, budget / 1e6)
    log.info("%s: giữ %d tài liệu, %.1f MB; bỏ ngắn=%d trùng=%d rò_rỉ=%d", lang, st["kept_docs"], kept_bytes / 1e6,
             st["drop_short_or_bad"], st["drop_duplicate"], st["drop_leak"])
    record_processed(f"mono/{lang}", "build_mono.py", [src.relative_to(DATA_DIR).as_posix()],
                     sorted(outdir.glob("*")), config_hash({"budget": budget, **m}), **{k: meta[k] for k in
                     ("budget_bytes", "kept_bytes", "shortfall", "leakage_filter", "stats")})
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", nargs="+")
    ap.add_argument("--scale", type=float, default=1.0, help="phải khớp --scale đã dùng khi fetch")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-leakage", action="store_true")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config()
    prem = load_premium()
    plan = byte_budgets(prem, cfg, args.scale)
    budgets = per_language_budget(prem, cfg, args.scale)
    langs = args.langs or [l for l in budgets if mono_raw_dir(l).exists()]
    if not langs:
        raise SystemExit("Không có raw nào để xử lý. Chạy fetch.py fineweb2 trước.")
    idx = None
    if not args.no_leakage:
        idx = load_or_none()
        if idx is None:
            raise SystemExit("Chưa có processed/leakage_index.npz. Chạy: python scripts/leakage_check.py build "
                             "(hoặc thêm --no-leakage nếu chỉ thử).")
        log.info("chỉ mục rò rỉ: %d %d-gram", idx.keys.size, idx.n)
    for l in langs:
        if l not in budgets:
            log.error("%s không có trong kế hoạch ngân sách", l)
            continue
        build_language(l, budgets[l], plan, cfg, idx, args)


if __name__ == "__main__":
    main()
