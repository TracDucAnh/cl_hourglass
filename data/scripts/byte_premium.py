"""Byte premium  prem_l = byte_l / byte_eng  đo trên FLORES devtest (UTF-8, sau NFKC).

    python scripts/byte_premium.py
Ra: processed/mono/byte_premium.json  (premium từng ngôn ngữ + kế hoạch ngân sách byte)
In ra bảng kế hoạch để bạn biết trước dung lượng cần tải/lưu.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from common import (PARALLEL, PREMIUM_PATH, atomic_write_text, byte_budgets, config_hash, load_config, log,
                    per_language_budget, read_jsonl, record_processed, setup_logging)


def compute(path: Path, base: str = "eng") -> dict:
    nbytes, nchars, ids = defaultdict(int), defaultdict(int), defaultdict(set)
    for r in read_jsonl(path):
        nbytes[r["lang"]] += len(r["text"].encode("utf-8"))
        nchars[r["lang"]] += len(r["text"])
        ids[r["lang"]].add(r["id"])
    if base not in nbytes:
        raise SystemExit(f"Không có ngôn ngữ cơ sở '{base}' trong {path}")
    sizes = {len(v) for v in ids.values()}
    if len(sizes) != 1:
        log.warning("số mục khác nhau giữa ngôn ngữ: %s", {l: len(v) for l, v in ids.items()})
    return {
        l: {"bytes": nbytes[l], "chars": nchars[l], "n_sentences": len(ids[l]), "premium": nbytes[l] / nbytes[base]}
        for l in nbytes
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=str(PARALLEL / "flores_devtest.jsonl"), help="mặc định: FLORES devtest")
    ap.add_argument("--base", default="eng")
    ap.add_argument("--out", default=str(PREMIUM_PATH))
    ap.add_argument("--scale", type=float, default=1.0, help="chỉ để in kế hoạch")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config()
    src = Path(args.file)
    if not src.exists():
        raise SystemExit(f"Thiếu {src}. Chạy: build_parallel.py flores")
    langs = compute(src, args.base)
    prem = {l: v["premium"] for l, v in langs.items()}
    plan = byte_budgets(prem, cfg, args.scale)
    per_lang = per_language_budget(prem, cfg, args.scale)
    out = {
        "source": src.name,
        "base": args.base,
        "params": {k: cfg["mono"][k] for k in ("T_tokens", "bytes_per_token_en", "raw_margin", "extra_bytes")},
        "languages": langs,
        "budget_plan_bytes": plan,
        "per_language_max_bytes": per_lang,
    }
    atomic_write_text(args.out, json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    record_processed("mono/byte_premium", "byte_premium.py", [src.name], [Path(args.out)],
                     config_hash(out["params"]), source=src.name)

    print(f"\nByte premium (cơ sở {args.base}, nguồn {src.name})")
    print(f"{'lang':5} {'bytes':>9} {'chars':>9} {'premium':>8}")
    for l, v in sorted(langs.items(), key=lambda kv: kv[1]["premium"]):
        print(f"{l:5} {v['bytes']:>9} {v['chars']:>9} {v['premium']:>8.3f}")
    print(f"\nKế hoạch ngân sách (T={float(cfg['mono']['T_tokens']):.2g} token, "
          f"{cfg['mono']['bytes_per_token_en']} B/token tiếng Anh, scale={args.scale}), đơn vị MB")
    for name, d in plan.items():
        print(f"  {name}: " + ", ".join(f"{l}={b / 1e6:,.0f}" for l, b in d.items()) + f"   (tổng {sum(d.values()) / 1e6:,.0f})")
    tot = sum(per_lang.values())
    print(f"\nByte cần giữ trong processed/mono: {tot / 1e9:.2f} GB; "
          f"tải raw (x{cfg['mono']['raw_margin']}): ~{tot * float(cfg['mono']['raw_margin']) / 1e9:.2f} GB chưa nén")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
