"""raw/ -> processed/tasks/  (XNLI, UPOS từ UD, Belebele)

    python scripts/build_tasks.py                # cả ba
    python scripts/build_tasks.py upos

Ra:
  tasks/xnli/train_eng.jsonl (10 000 mẫu, seed cố định), validation_{lang}.jsonl, test_{lang}.jsonl (5 010 mẫu)
        {id, lang, idx, premise, hypothesis, label, label_name}   idx = vị trí căn song song giữa ngôn ngữ
  tasks/upos/{lang}.jsonl      {id, lang, split, sent_id, text, tokens, upos}
  tasks/belebele/{lang}.jsonl  200 câu hỏi con, cùng bộ câu hỏi ở mọi ngôn ngữ
        {id, lang, link, question_number, passage, question, answers[4], correct (0..3)}
"""
from __future__ import annotations

import argparse
import random
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterator, List

from clean import normalize_sentence
from common import RAW, TASKS, config_hash, flores_code, load_config, log, read_jsonl, record_processed, setup_logging, write_jsonl

LABELS = ["entailment", "neutral", "contradiction"]


# ------------------------------------------------------------------ XNLI
def _xnli_split_files(iso: str, split: str) -> List[Path]:
    return sorted((RAW / "xnli" / iso).glob(f"{split}-*.parquet"))


def _xnli_rows(files: List[Path], lang: str, split: str) -> Iterator[dict]:
    import pandas as pd

    i = 0
    for f in files:
        df = pd.read_parquet(f)
        for r in df.itertuples(index=False):
            lab = int(r.label)
            yield {"id": f"xnli_{split}_{lang}_{i:06d}", "lang": lang, "idx": i,
                   "premise": normalize_sentence(str(r.premise)), "hypothesis": normalize_sentence(str(r.hypothesis)),
                   "label": lab, "label_name": LABELS[lab] if 0 <= lab < 3 else None}
            i += 1


def build_xnli(cfg, langs: List[str]) -> None:
    out = TASKS / "xnli"
    size = int(cfg["sources"]["xnli_train_size"])
    for l in langs:
        iso = cfg["languages"][l].get("xnli")
        if not iso:
            log.warning("XNLI: %s không có, bỏ qua", l)
            continue
        for split in ("validation", "test"):
            files = _xnli_split_files(iso, split)
            if not files:
                raise SystemExit(f"Thiếu raw/xnli/{iso}/{split}-*.parquet. Chạy: fetch.py xnli")
            path = out / f"{split}_{l}.jsonl"
            n = write_jsonl(path, _xnli_rows(files, l, split))
            if split == "test" and n != 5010:
                log.warning("XNLI test %s: %d mẫu (kỳ vọng 5010)", l, n)
            log.info("xnli %-10s %s: %d mẫu", split, l, n)
            record_processed(f"tasks/xnli/{split}_{l}", "build_tasks.py", [f"raw/xnli/{iso}"], [path],
                             config_hash({"split": split}), n_rows=n)
        if l == "eng":
            files = _xnli_split_files(iso, "train")
            if not files:
                raise SystemExit("Thiếu raw/xnli/en/train-*.parquet")
            rows = list(_xnli_rows(files, l, "train"))
            rng = random.Random(cfg["seed"])
            pick = sorted(rng.sample(range(len(rows)), min(size, len(rows))))
            sub = []
            for k, j in enumerate(pick):
                r = dict(rows[j])
                r["id"], r["idx"] = f"xnli_train_eng_{k:06d}", k
                sub.append(r)
            path = out / "train_eng.jsonl"
            n = write_jsonl(path, sub)
            log.info("xnli train eng: %d / %d mẫu (seed=%s)", n, len(rows), cfg["seed"])
            record_processed("tasks/xnli/train_eng", "build_tasks.py", ["raw/xnli/en"], [path],
                             config_hash({"seed": cfg["seed"], "size": size}), n_rows=n, source_rows=len(rows))


# ------------------------------------------------------------------ UD / UPOS
def parse_conllu(path: Path) -> Iterator[dict]:
    sent_id, text, toks, ups = None, None, [], []

    def flush():
        nonlocal sent_id, text, toks, ups
        if toks:
            yield_row = {"sent_id": sent_id, "text": text, "tokens": toks, "upos": ups}
            toks, ups, sent_id, text = [], [], None, None
            return yield_row
        sent_id, text = None, None
        return None

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                r = flush()
                if r:
                    yield r
            elif line.startswith("#"):
                if line.startswith("# sent_id"):
                    sent_id = line.split("=", 1)[1].strip()
                elif line.startswith("# text ="):
                    text = line.split("=", 1)[1].strip()
            else:
                c = line.split("\t")
                if len(c) < 4 or "-" in c[0] or "." in c[0]:  # bỏ token đa từ và nút rỗng
                    continue
                toks.append(c[1])
                ups.append(c[3])
    r = flush()
    if r:
        yield r


def build_upos(cfg, langs: List[str]) -> None:
    out = TASKS / "upos"
    for l in langs:
        repo = cfg["languages"][l].get("ud")
        if not repo:
            log.warning("UPOS: %s không có treebank, bỏ qua", l)
            continue
        files = sorted(f for f in (RAW / "ud" / repo).glob("*.conllu") if re.search(r"-ud-(train|dev|test)\.conllu$", f.name))
        if not files:
            raise SystemExit(f"Thiếu raw/ud/{repo}/*.conllu. Chạy: fetch.py ud")

        def gen():
            k = 0
            for f in files:
                split = re.search(r"-ud-(train|dev|test)\.conllu$", f.name).group(1)
                for r in parse_conllu(f):
                    toks = [unicodedata.normalize("NFKC", t) for t in r["tokens"]]
                    if any(not t.strip() for t in toks):
                        continue
                    text = normalize_sentence(r["text"] or " ".join(toks))
                    yield {"id": f"upos_{l}_{k:06d}", "lang": l, "split": split, "sent_id": r["sent_id"],
                           "text": text, "tokens": toks, "upos": r["upos"]}
                    k += 1

        path = out / f"{l}.jsonl"
        n = write_jsonl(path, gen())
        log.info("upos %s (%s): %d câu", l, repo, n)
        record_processed(f"tasks/upos/{l}", "build_tasks.py", [f"raw/ud/{repo}"], [path], config_hash({"repo": repo}),
                         n_rows=n, treebank=repo, ud_ref=cfg["sources"]["ud_ref"])


# ------------------------------------------------------------------ Belebele
def _belebele_file(code: str) -> Path | None:
    for p in sorted((RAW / "belebele").rglob(f"*{code}*")):
        if p.is_file() and p.suffix in (".jsonl", ".json", ".parquet"):
            return p
    return None


def _belebele_rows(p: Path) -> List[dict]:
    if p.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(p).to_dict("records")
    return list(read_jsonl(p))


def build_belebele(cfg, langs: List[str]) -> None:
    out = TASKS / "belebele"
    nq = int(cfg["sources"]["belebele_questions"])
    data: Dict[str, Dict[tuple, dict]] = {}
    for l in langs:
        code = flores_code(cfg, l)
        p = _belebele_file(code)
        if p is None:
            raise SystemExit(f"Thiếu dữ liệu Belebele cho {code}. Chạy: fetch.py belebele")
        data[l] = {(r["link"], int(r["question_number"])): r for r in _belebele_rows(p)}
    common_keys = sorted(set.intersection(*(set(d) for d in data.values())))
    rng = random.Random(cfg["seed"])
    pick = sorted(rng.sample(common_keys, min(nq, len(common_keys))))
    log.info("belebele: %d khóa chung, chọn %d (seed=%s)", len(common_keys), len(pick), cfg["seed"])
    for l in langs:
        def gen():
            for k, key in enumerate(pick):
                r = data[l][key]
                yield {"id": f"belebele_{k:04d}", "lang": l, "link": key[0], "question_number": key[1],
                       "passage": normalize_sentence(r["flores_passage"]), "question": normalize_sentence(r["question"]),
                       "answers": [normalize_sentence(str(r[f"mc_answer{i}"])) for i in range(1, 5)],
                       "correct": int(r["correct_answer_num"]) - 1}

        path = out / f"{l}.jsonl"
        n = write_jsonl(path, gen())
        record_processed(f"tasks/belebele/{l}", "build_tasks.py", ["raw/belebele"], [path],
                         config_hash({"seed": cfg["seed"], "n": nq}), n_rows=n)
    log.info("belebele: ghi %d ngôn ngữ", len(langs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", help="xnli | upos | belebele (mặc định: cả ba)")
    ap.add_argument("--langs", nargs="+")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config()
    langs = args.langs or cfg["study_languages"]
    targets = args.targets or ["xnli", "upos", "belebele"]
    fn = {"xnli": build_xnli, "upos": build_upos, "belebele": build_belebele}
    for t in targets:
        if t not in fn:
            raise SystemExit(f"Mục tiêu không hợp lệ: {t}")
        fn[t](cfg, langs)


if __name__ == "__main__":
    main()
