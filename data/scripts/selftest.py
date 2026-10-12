"""Kiểm thử toàn bộ pipeline OFFLINE trên dữ liệu giả (không đụng tới data/ thật).

    python scripts/selftest.py            # ~1 phút
    python scripts/selftest.py --keep     # giữ thư mục tạm để xem

Tạo một bản sao data/ trong thư mục tạm, dựng raw/ giả (FLORES tar, XNLI parquet, Belebele jsonl,
FineWeb jsonl.gz có tài liệu trùng / ngắn / rò rỉ), chạy các script theo thứ tự và kiểm tra kết quả.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC_DATA = HERE.parent

ALPHA = {
    "latin": "abcdefghijklmnopqrstuvwxyz",
    "cyr": "абвгдежзиклмнопрстуфхцчшщыэюя",
    "dev": "कखगघचछजझटठडढणतथदधनपफबभमयरलवशसह",
    "arb": "ابتثجحخدذرزسشصضطظعغفقكلمنهوي",
    "cjk": "我你他们的一是不了人有来这中大为上个国说到要出会可也你对生能而子那得于着下自之年过发后作里",
    "tha": "กขคงจฉชซญดตถทนบปผฝพฟภมยรลวศษสหอ",
}
LANG_ALPHA = {"eng": "latin", "fra": "latin", "rus": "cyr", "hin": "dev", "zho": "cjk", "vie": "latin", "tur": "latin",
              "arb": "arb", "jpn": "cjk", "fin": "latin", "swh": "latin", "tha": "tha"}
MONO8 = ["fra", "zho", "rus", "tur", "hin", "arb", "vie", "eng"]


def make_vocab(lang: str, rng: random.Random, n: int = 400):
    al = ALPHA[LANG_ALPHA[lang]]
    if LANG_ALPHA[lang] == "cjk":
        return [rng.choice(al) + rng.choice(al) for _ in range(n)]
    return ["".join(rng.choice(al) for _ in range(rng.randint(2, 9))) for _ in range(n)]


def zipf_words(vocab, k, rng):
    w = [1.0 / (i + 1) for i in range(len(vocab))]
    return rng.choices(vocab, weights=w, k=k)


def sentence(lang, vocab, rng, n=22):
    ws = zipf_words(vocab, n, rng)
    return ("" if LANG_ALPHA[lang] == "cjk" else " ").join(ws)


class T:
    ok = 0

    @staticmethod
    def check(cond, msg):
        if not cond:
            raise AssertionError("THẤT BẠI: " + msg)
        T.ok += 1
        print(f"  ok  {msg}")


def run(args, env):
    r = subprocess.run([sys.executable, *map(str, args)], env=env, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-3000:])
        raise SystemExit(f"Lệnh lỗi: {args}")
    return r.stdout + r.stderr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    import pandas as pd
    import yaml

    tmp = Path(tempfile.mkdtemp(prefix="hourglass_selftest_"))
    print("thư mục tạm:", tmp)
    shutil.copytree(SRC_DATA, tmp / "data", ignore=shutil.ignore_patterns("raw", "processed", "manifest.json", "__pycache__", "*.zip"))
    d = tmp / "data"
    for sub in ("raw", "processed"):
        (d / sub).mkdir()
    (d / "processed" / "mono").mkdir(parents=True)
    cfg = yaml.safe_load(open(d / "config.yaml", encoding="utf-8"))
    cfg["mono"].update({"T_tokens": 2.0e5, "bytes_per_token_en": 4.0, "extra_bytes": 100000, "shard_bytes": 60000})
    cfg["sources"].update({"xnli_train_size": 100, "belebele_questions": 10})
    yaml.safe_dump(cfg, open(d / "config.yaml", "w", encoding="utf-8"), allow_unicode=True)
    (d / "manifest.json").write_text(json.dumps({"schema_version": 1, "raw": {}, "processed": {}, "verification": {}}))
    env = {**os.environ, "HOURGLASS_DATA_DIR": str(d), "PYTHONIOENCODING": "utf-8"}
    os.environ["HOURGLASS_DATA_DIR"] = str(d)
    sys.path.insert(0, str(HERE))
    S = HERE
    rng = random.Random(0)
    langs12 = cfg["study_languages"]
    vocab = {l: make_vocab(l, rng) for l in langs12}

    # ---------------- 1. FLORES giả -> tar.gz -> extract_flores_tar
    import fetch  # noqa: E402

    print("\n[1] FLORES (tar giả) + build_parallel")
    sizes = {"dev": 40, "devtest": 50}
    flores_sent = {}
    tarp = tmp / "flores200_dataset.tar.gz"
    with tarfile.open(tarp, "w:gz") as tf:
        for split, n in sizes.items():
            for l in langs12:
                code = cfg["languages"][l]["flores"]
                lines = [sentence(l, vocab[l], rng) for _ in range(n)]
                flores_sent[(split, l)] = lines
                b = ("\n".join(lines) + "\n").encode("utf-8")
                ti = tarfile.TarInfo(f"flores200_dataset/{split}/{code}.{split}")
                ti.size = len(b)
                tf.addfile(ti, io.BytesIO(b))
    n = fetch.extract_flores_tar(tarp, d / "raw" / "flores", [cfg["languages"][l]["flores"] for l in langs12])
    T.check(n == 24, f"extract_flores_tar rút đủ 24 tệp (dev+devtest x 12): {n}")
    out = run([S / "build_parallel.py", "flores"], env)
    rows = [json.loads(x) for x in open(d / "processed/parallel/flores_devtest.jsonl", encoding="utf-8")]
    T.check(len(rows) == 50 * 12, f"flores_devtest có {len(rows)} dòng = 50 x 12")
    T.check({"id", "lang", "text"} == set(rows[0]), "schema {id, lang, text}")

    # ---------------- 2. UD zip: chỉ lấy tệp ở gốc
    print("\n[2] UD: chỉ lấy *-ud-(train|dev|test).conllu ở gốc repo")
    zp = tmp / "ud.zip"
    conllu = "# sent_id = s1\n# text = Hello world .\n1\tHello\thello\tINTJ\t_\t_\t0\troot\t_\t_\n2\tworld\tworld\tNOUN\t_\t_\t1\tdep\t_\t_\n3-4\tx\t_\t_\t_\t_\t_\t_\t_\t_\n\n"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("UD_X-r2.18/en_x-ud-train.conllu", conllu)
        z.writestr("UD_X-r2.18/not-to-release/extra.conllu", conllu)
        z.writestr("UD_X-r2.18/NewTest.conllu", conllu)
        z.writestr("UD_X-r2.18/README.md", "r")
    names = fetch.extract_ud_zip(zp, tmp / "udout")
    T.check(sorted(names) == ["README.md", "en_x-ud-train.conllu"], f"chỉ giữ tệp hợp lệ: {sorted(names)}")
    import build_tasks

    parsed = list(build_tasks.parse_conllu(tmp / "udout" / "en_x-ud-train.conllu"))
    T.check(parsed[0]["tokens"] == ["Hello", "world"] and parsed[0]["upos"] == ["INTJ", "NOUN"], "parse_conllu bỏ token đa từ")

    # ---------------- 3. byte premium
    print("\n[3] byte_premium")
    run([S / "byte_premium.py"], env)
    prem = json.load(open(d / "processed/mono/byte_premium.json", encoding="utf-8"))
    T.check(abs(prem["languages"]["eng"]["premium"] - 1.0) < 1e-12, "premium(eng) = 1")
    T.check(prem["languages"]["hin"]["premium"] > 1.5 and prem["languages"]["zho"]["premium"] < 2.5, "premium hợp lý theo bảng chữ")
    plan = prem["budget_plan_bytes"]
    exp = int(round(2e5 / 2 * 4 * prem["languages"]["fra"]["premium"]))
    T.check(plan["L2"]["fra"] == exp, f"ngân sách L2/fra = (T/L)*4*prem = {exp}")

    # ---------------- 4. XNLI + Belebele giả
    print("\n[4] XNLI / Belebele (parquet / jsonl giả) + build_tasks")
    xl = {"eng": "en", "fra": "fr", "rus": "ru", "hin": "hi", "zho": "zh", "vie": "vi", "tur": "tr", "arb": "ar", "swh": "sw", "tha": "th"}
    for l, iso in xl.items():
        (d / "raw/xnli" / iso).mkdir(parents=True)
        splits = {"validation": 20, "test": 30}
        if l == "eng":
            splits["train"] = 300
        for sp, n in splits.items():
            pd.DataFrame({"premise": [sentence(l, vocab[l], rng, 8) + "  " for _ in range(n)],
                          "hypothesis": [sentence(l, vocab[l], rng, 5) for _ in range(n)],
                          "label": [rng.randrange(3) for _ in range(n)]}).to_parquet(d / f"raw/xnli/{iso}/{sp}-00000-of-00001.parquet")
    (d / "raw/belebele/data").mkdir(parents=True)
    keys = [(f"https://x/{i // 3}", i % 3 + 1) for i in range(40)]
    for l in langs12:
        with open(d / f"raw/belebele/data/{cfg['languages'][l]['flores']}.jsonl", "w", encoding="utf-8") as f:
            for link, q in keys:
                f.write(json.dumps({"link": link, "question_number": q, "flores_passage": sentence(l, vocab[l], rng, 15),
                                    "question": sentence(l, vocab[l], rng, 6), "mc_answer1": "a", "mc_answer2": "b",
                                    "mc_answer3": "c", "mc_answer4": "d", "correct_answer_num": str(rng.randint(1, 4))},
                                   ensure_ascii=False) + "\n")
    run([S / "build_tasks.py", "xnli", "belebele"], env)
    tr = [json.loads(x) for x in open(d / "processed/tasks/xnli/train_eng.jsonl", encoding="utf-8")]
    T.check(len(tr) == 100, "XNLI train_eng lấy mẫu 100 (theo cấu hình thử)")
    T.check(not tr[0]["premise"].endswith(" "), "XNLI chuẩn hóa khoảng trắng")
    T.check(not (d / "processed/tasks/xnli/test_jpn.jsonl").exists() and not (d / "processed/tasks/xnli/test_fin.jsonl").exists(),
            "XNLI không có jpn/fin (đúng đặc tả)")
    b1 = [json.loads(x)["link"] for x in open(d / "processed/tasks/belebele/fra.jsonl", encoding="utf-8")]
    b2 = [json.loads(x)["link"] for x in open(d / "processed/tasks/belebele/tha.jsonl", encoding="utf-8")]
    T.check(len(b1) == 10 and b1 == b2, "Belebele: cùng 10 câu hỏi ở mọi ngôn ngữ")

    # ---------------- 5. raw FineWeb giả (có trùng, ngắn, rò rỉ)
    print("\n[5] FineWeb giả -> leakage index -> build_mono")
    leak_sentence = flores_sent[("devtest", "fra")][3]
    for l in MONO8:
        rd = d / "raw" / ("fineweb" if l == "eng" else "fineweb2") / l
        rd.mkdir(parents=True)
        docs = []
        for i in range(1200):
            docs.append({"id": f"{l}{i}", "text": "\n\n".join(sentence(l, vocab[l], rng, 30) for _ in range(rng.randint(3, 6)))})
        docs.insert(8, dict(docs[7], id="dup"))              # trùng chính xác (ngay sau bản gốc)
        docs.insert(5, {"id": "short", "text": "ngắn"})      # quá ngắn
        if l == "fra":
            docs.insert(10, {"id": "leak", "text": "mở đầu " + sentence(l, vocab[l], rng, 15) + " " + leak_sentence + " kết thúc " + sentence(l, vocab[l], rng, 15)})
        with gzip.open(rd / "part_00000.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in docs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    run([S / "leakage_check.py", "build", "--sources", "flores", "xnli"], env)
    out = run([S / "build_mono.py"], env)
    meta = {l: json.load(open(d / f"processed/mono/{l}/_meta.json", encoding="utf-8")) for l in MONO8}
    T.check(meta["fra"]["stats"]["drop_leak"] == 1, "đúng 1 tài liệu fra bị loại vì rò rỉ FLORES")
    T.check(meta["fra"]["stats"]["drop_duplicate"] == 1 and meta["fra"]["stats"]["drop_short_or_bad"] == 1, "loại 1 trùng + 1 ngắn")
    for l in MONO8:
        m = meta[l]
        T.check(not m["shortfall"] and m["budget_bytes"] <= m["kept_bytes"] < m["budget_bytes"] + 5000,
                f"{l}: giữ {m['kept_bytes']} byte, ngân sách {m['budget_bytes']}")
    T.check(meta["fra"]["n_shards"] > 1, "shard xoay vòng (fra có nhiều shard)")
    T.check(meta["fra"]["budget_per_config_bytes"].keys() == {"L2", "L4", "L8"}, "fra thuộc cả L2, L4, L8")
    T.check(meta["fra"]["budget_bytes"] == plan["L2"]["fra"], "ngân sách giữ lại = max qua cấu hình = L2")
    T.check(meta["eng"]["budget_bytes"] == plan["L8"]["eng"], "eng chỉ ở L8")
    blob = "".join(open(p, encoding="utf-8").read() for p in sorted((d / "processed/mono/fra").glob("shard_*.jsonl")))
    T.check(leak_sentence not in blob, "câu FLORES không còn trong shard fra")
    T.check(all(len(json.loads(l)["id"]) for l in open(d / "processed/mono/fra/shard_00000.jsonl", encoding="utf-8")), "schema shard có id")

    # leakage CLI
    out = run([S / "leakage_check.py", "check", d / "processed/parallel/flores_devtest.jsonl"], env)
    T.check("100.000%" in out, "check: FLORES tự phát hiện 100% (n-gram >=13 đơn vị)")

    # ---------------- 6. wordfreq + lang2vec
    print("\n[6] typology")
    run([S / "build_typology.py", "wordfreq", "--langs", "fra", "zho"], env)
    lines = open(d / "processed/typology/wordfreq/fra.tsv", encoding="utf-8").read().splitlines()
    c = [int(x.split("\t")[2]) for x in lines[1:]]
    T.check(lines[0] == "rank\tword\tcount" and c == sorted(c, reverse=True) and len(c) > 50, "wordfreq fra: sắp giảm dần theo hạng")
    zm = json.load(open(d / "processed/typology/wordfreq/zho.meta.json", encoding="utf-8"))
    T.check(zm["segmenter"] in ("jieba", "char-fallback"), f"zho segmenter = {zm['segmenter']}")
    run([S / "build_typology.py", "lang2vec"], env)
    T.check((d / "processed/typology/lang2vec_distance_syntactic.csv").exists(), "lang2vec: có ma trận khoảng cách")

    # ---------------- 7. manifest
    print("\n[7] manifest")
    m = json.load(open(d / "manifest.json", encoding="utf-8"))
    T.check({"flores_devtest", "flores_dev"} <= {k.split("/")[1] for k in m["processed"] if k.startswith("parallel/")}, "manifest ghi parallel/flores_*")
    T.check("mono/fra" in m["processed"] and all("sha256" in v for v in m["processed"]["mono/fra"]["files"].values()), "manifest có checksum shard")
    T.check("leakage_index" in m["processed"], "manifest ghi leakage_index")

    print(f"\nTẤT CẢ {T.ok} KIỂM TRA ĐỀU QUA.")
    if a.keep:
        print("giữ lại:", tmp)
    else:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
