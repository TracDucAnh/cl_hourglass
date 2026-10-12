"""processed/typology/: đặc trưng loại hình (lang2vec, WALS) và bảng tần suất từ.

    python scripts/build_typology.py lang2vec
    python scripts/build_typology.py wals          # cần raw/wals (fetch.py wals)
    python scripts/build_typology.py wordfreq      # cần processed/mono/{lang}/ (build_mono.py)
    python scripts/build_typology.py               # lang2vec + wals

Khoảng cách lang2vec: lang2vec 1.1.x không còn hàm distance(), nên ta tự tính khoảng cách góc
(arccos(cosine)/pi, như URIEL) trên các vector *_knn (đã điền đủ), fam, geo.
  syntactic = syntax_knn, phonological = phonology_knn, inventory = inventory_knn,
  genetic = fam, geographic = geo, featural = nối syntax+phonology+inventory (knn).

Tần suất từ: tách từ bằng regex Unicode; zho dùng jieba, jpn dùng janome, tha dùng pythainlp
(nếu chưa cài thì lùi về đếm ký tự và ghi rõ trong meta). Tiếng Việt: đơn vị là âm tiết
(tách theo khoảng trắng), KHÔNG phải từ ghép.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import regex

from common import (MONO, RAW, TYPOLOGY, atomic_write_text, config_hash, load_config, log, read_jsonl, record_processed,
                    setup_logging)

L2V_SETS = {"syntactic": "syntax_knn", "phonological": "phonology_knn", "inventory": "inventory_knn",
            "genetic": "fam", "geographic": "geo"}


def _write_matrix(path: Path, langs: List[str], m: np.ndarray) -> None:
    lines = [",".join([""] + langs)] + [",".join([l] + [f"{v:.6f}" for v in row]) for l, row in zip(langs, m)]
    atomic_write_text(path, "\n".join(lines) + "\n")


def angular_distance_matrix(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    xn = x / n
    cos = np.clip(xn @ xn.T, -1.0, 1.0)
    d = np.arccos(cos) / np.pi
    np.fill_diagonal(d, 0.0)
    return d


# ------------------------------------------------------------------ lang2vec
def build_lang2vec(cfg) -> None:
    import lang2vec.lang2vec as l2v

    langs = cfg["study_languages"]
    codes = [cfg["languages"][l]["lang2vec"] for l in langs]
    avail = set(l2v.available_languages())
    miss = [c for c in codes if c not in avail]
    if miss:
        raise SystemExit(f"lang2vec không có mã: {miss} (sửa 'lang2vec' trong config.yaml)")
    TYPOLOGY.mkdir(parents=True, exist_ok=True)
    feats: Dict[str, np.ndarray] = {}
    paths: List[Path] = []
    for name, fs in L2V_SETS.items():
        r = l2v.get_features(codes, fs, header=True)
        names = [str(x) for x in r["CODE"]]
        mat = np.array([[float(v) for v in r[c]] for c in codes], dtype=float)
        if np.isnan(mat).any():
            raise SystemExit(f"{fs}: có giá trị thiếu; chọn bộ đặc trưng khác")
        feats[name] = mat
        p = TYPOLOGY / f"lang2vec_features_{fs}.csv"
        with open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["lang"] + names)
            for l, row in zip(langs, mat):
                w.writerow([l] + [f"{v:g}" for v in row])
        paths.append(p)
    feats["featural"] = np.concatenate([feats["syntactic"], feats["phonological"], feats["inventory"]], axis=1)
    for name, mat in feats.items():
        p = TYPOLOGY / f"lang2vec_distance_{name}.csv"
        _write_matrix(p, langs, angular_distance_matrix(mat))
        paths.append(p)
    log.info("lang2vec: %d bộ đặc trưng + %d ma trận khoảng cách", len(L2V_SETS), len(feats))
    record_processed("typology/lang2vec", "build_typology.py", ["pip:lang2vec"], paths, config_hash(L2V_SETS),
                     languages=langs, codes=codes, distance="angular (arccos cosine / pi), tự tính")


# ------------------------------------------------------------------ WALS
def build_wals(cfg) -> None:
    d = RAW / "wals" / "cldf"
    for fn in ("languages.csv", "parameters.csv", "values.csv"):
        if not (d / fn).exists():
            raise SystemExit(f"Thiếu {d / fn}. Chạy: python scripts/fetch.py wals")

    def rd(fn):
        with open(d / fn, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    langs_rows, params, values = rd("languages.csv"), rd("parameters.csv"), rd("values.csv")
    count: Counter = Counter(v["Language_ID"] for v in values)
    by_iso: Dict[str, List[dict]] = {}
    for r in langs_rows:
        by_iso.setdefault(r["ISO639P3code"], []).append(r)

    study = cfg["study_languages"]
    chosen: Dict[str, dict] = {}
    for l in study:
        iso = cfg["languages"][l]["wals"]
        cands = by_iso.get(iso, [])
        if not cands:
            log.warning("WALS: không có ngôn ngữ ISO %s (%s)", iso, l)
            continue
        best = max(cands, key=lambda r: count[r["ID"]])
        if len(cands) > 1:
            log.info("WALS %s: %d ứng viên %s, chọn %s", l, len(cands), [c["ID"] for c in cands], best["ID"])
        chosen[l] = best
        if count[best["ID"]] < 80:
            log.warning("WALS %s (%s): chỉ có %d đặc trưng -> khoảng cách WALS của ngôn ngữ này kém tin cậy; "
                        "ưu tiên khoảng cách lang2vec", l, best["ID"], count[best["ID"]])
    pid = [p["ID"] for p in params]
    val = {(v["Language_ID"], v["Parameter_ID"]): v["Value"] for v in values}
    TYPOLOGY.mkdir(parents=True, exist_ok=True)

    p_feat = TYPOLOGY / "wals_features.csv"
    with open(p_feat, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lang", "wals_id", "name", "family", "genus"] + pid)
        for l, r in chosen.items():
            w.writerow([l, r["ID"], r["Name"], r["Family"], r["Genus"]] + [val.get((r["ID"], p), "") for p in pid])
    p_par = TYPOLOGY / "wals_parameters.csv"
    with open(p_par, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter_id", "name"])
        for p in params:
            w.writerow([p["ID"], p["Name"]])

    ls = [l for l in study if l in chosen]
    n = len(ls)
    ham, shared = np.zeros((n, n)), np.zeros((n, n), dtype=int)
    for i, a in enumerate(ls):
        for j, b in enumerate(ls):
            ks = [p for p in pid if (chosen[a]["ID"], p) in val and (chosen[b]["ID"], p) in val]
            shared[i, j] = len(ks)
            if ks:
                ham[i, j] = sum(val[(chosen[a]["ID"], p)] != val[(chosen[b]["ID"], p)] for p in ks) / len(ks)
            else:
                ham[i, j] = np.nan
    p_d, p_n = TYPOLOGY / "wals_distance_hamming.csv", TYPOLOGY / "wals_shared_features.csv"
    _write_matrix(p_d, ls, ham)
    atomic_write_text(p_n, ",".join([""] + ls) + "\n" + "\n".join(",".join([l] + [str(v) for v in row]) for l, row in zip(ls, shared)) + "\n")
    log.info("WALS: %d ngôn ngữ; đặc trưng chung trung bình %.0f", n, shared[np.triu_indices(n, 1)].mean() if n > 1 else 0)
    record_processed("typology/wals", "build_typology.py", ["raw/wals"], [p_feat, p_par, p_d, p_n],
                     config_hash({"langs": ls}), matched={l: chosen[l]["ID"] for l in ls},
                     note="khoảng cách = tỉ lệ đặc trưng khác nhau trên các đặc trưng cả hai cùng có")


# ------------------------------------------------------------------ tần suất từ
_WORD = regex.compile(r"[\p{L}\p{N}\p{M}]+(?:['’\-][\p{L}\p{N}\p{M}]+)*")
_HAS_LETTER = regex.compile(r"\p{L}")


def _regex_tokens(text: str) -> List[str]:
    return _WORD.findall(text)


def get_segmenter(lang: str) -> tuple[Callable[[str], List[str]], str]:
    if lang == "zho":
        try:
            import jieba

            jieba.setLogLevel(60)
            return (lambda t: jieba.lcut(t, HMM=True)), "jieba"
        except ImportError:
            pass
    elif lang == "jpn":
        try:
            from janome.tokenizer import Tokenizer

            tk = Tokenizer()
            return (lambda t: [x for x in tk.tokenize(t, wakati=True)]), "janome"
        except ImportError:
            pass
    elif lang == "tha":
        try:
            from pythainlp.tokenize import word_tokenize

            return (lambda t: word_tokenize(t, engine="newmm", keep_whitespace=False)), "pythainlp-newmm"
        except ImportError:
            pass
    else:
        return _regex_tokens, "regex-syllable" if lang == "vie" else "regex"
    log.warning("%s: thiếu bộ tách từ -> đếm theo KÝ TỰ (cài jieba / janome / pythainlp để có từ)", lang)
    return (lambda t: [c for c in t if _HAS_LETTER.match(c)]), "char-fallback"


def build_wordfreq(cfg, langs: List[str], max_bytes: int, min_count: int) -> None:
    out = TYPOLOGY / "wordfreq"
    out.mkdir(parents=True, exist_ok=True)
    for l in langs:
        shards = sorted((MONO / l).glob("shard_*.jsonl"))
        if not shards:
            log.warning("%s: chưa có processed/mono/%s (chạy build_mono.py); bỏ qua", l, l)
            continue
        seg, seg_name = get_segmenter(l)
        cnt: Counter = Counter()
        used = docs = 0
        for sh in shards:
            for r in read_jsonl(sh):
                t = r["text"]
                used += len(t.encode("utf-8"))
                docs += 1
                for tok in seg(t):
                    tok = tok.lower()
                    if _HAS_LETTER.search(tok):
                        cnt[tok] += 1
                if used >= max_bytes:
                    break
            if used >= max_bytes:
                break
        total = sum(cnt.values())
        items = [(w, c) for w, c in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])) if c >= min_count]
        p = out / f"{l}.tsv"
        lines = ["rank\tword\tcount"] + [f"{i}\t{w}\t{c}" for i, (w, c) in enumerate(items, 1)]
        atomic_write_text(p, "\n".join(lines) + "\n")
        meta = {"lang": l, "segmenter": seg_name, "bytes_used": used, "docs": docs, "tokens": total,
                "types_total": len(cnt), "types_written": len(items), "min_count": min_count}
        atomic_write_text(out / f"{l}.meta.json", json.dumps(meta, indent=2) + "\n")
        log.info("wordfreq %s [%s]: %d token, %d kiểu từ", l, seg_name, total, len(cnt))
        record_processed(f"typology/wordfreq/{l}", "build_typology.py", [f"processed/mono/{l}"], [p],
                         config_hash({"max_bytes": max_bytes, "min_count": min_count}), **meta)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", help="lang2vec | wals | wordfreq (mặc định: lang2vec wals)")
    ap.add_argument("--langs", nargs="+", help="chỉ cho wordfreq")
    ap.add_argument("--max-bytes", type=int, default=50_000_000, help="byte văn bản tối đa mỗi ngôn ngữ khi đếm tần suất")
    ap.add_argument("--min-count", type=int, default=1)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config()
    for t in args.targets or ["lang2vec", "wals"]:
        if t == "lang2vec":
            build_lang2vec(cfg)
        elif t == "wals":
            build_wals(cfg)
        elif t == "wordfreq":
            build_wordfreq(cfg, args.langs or cfg["wordfreq_languages"], args.max_bytes, args.min_count)
        else:
            raise SystemExit(f"Mục tiêu không hợp lệ: {t}")


if __name__ == "__main__":
    main()
