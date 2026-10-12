"""Tải dữ liệu thô vào data/raw/ và ghi nguồn + checksum vào manifest.json.

Ví dụ
    python scripts/fetch.py --verify                 # kiểm tra nguồn [XM] còn sống, KHÔNG tải
    python scripts/fetch.py                          # nhóm 'small' (nhẹ, vài trăm MB)
    python scripts/fetch.py flores ntrex             # chỉ vài bộ
    python scripts/fetch.py fineweb2 --scale 0.01    # nhóm lớn; bắt buộc có byte_premium.json
    python scripts/fetch.py all

Nhóm: small = flores ntrex xnli ud belebele muse glotlid wals
      large = fineweb2 (kèm FineWeb tiếng Anh) opus100
Biến môi trường: HF_TOKEN (cần cho flores_plus; nên đặt để tải HF nhanh hơn).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List

from common import (RAW, ShardWriter, config_hash, download, load_config, load_premium, log, per_language_budget,
                    record_raw, setup_logging, sha256_file, update_manifest, url_ok, utcnow)

SMALL = ["flores", "ntrex", "xnli", "ud", "belebele", "muse", "glotlid", "wals"]
LARGE = ["fineweb2", "opus100"]


def _hf_token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None


def _langs(args, cfg, default=None) -> List[str]:
    langs = args.langs or default or cfg["study_languages"]
    bad = [l for l in langs if l not in cfg["languages"]]
    if bad:
        raise SystemExit(f"Ngôn ngữ không có trong config.yaml: {bad}")
    return langs


# ------------------------------------------------------------------ FLORES
def extract_flores_tar(tgz: Path, out: Path, codes: List[str]) -> int:
    """Rút dev/devtest của các mã đã chọn từ flores200_dataset.tar.gz -> out/{split}/{code}.{split}."""
    want = {(s, f"{c}.{s}") for s in ("dev", "devtest") for c in codes}
    got = set()
    with tarfile.open(tgz, "r:gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            parts = Path(m.name).parts
            if len(parts) < 2 or (parts[-2], parts[-1]) not in want:
                continue
            dest = out / parts[-2] / parts[-1]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(tf.extractfile(m).read())
            got.add((parts[-2], parts[-1]))
    missing = want - got
    if missing:
        raise RuntimeError(f"Thiếu trong tar: {sorted(missing)[:6]} ...")
    return len(got)


def fetch_flores_plus(args, cfg, out: Path, langs: List[str]) -> None:
    """Phương án dự phòng (chưa kiểm chứng cấu trúc repo; [XM]). Cần HF_TOKEN + đã chấp nhận điều khoản."""
    import pandas as pd
    from huggingface_hub import snapshot_download

    tmp = out / "_flores_plus_hf"
    codes = {l: cfg["languages"][l].get("flores_plus", cfg["languages"][l]["flores"]) for l in langs}
    snapshot_download(cfg["sources"]["flores_plus_repo"], repo_type="dataset", local_dir=tmp,
                      allow_patterns=[f"*{c}*" for c in codes.values()], token=_hf_token())
    for l in langs:
        for split in ("dev", "devtest"):
            cands = [p for p in tmp.rglob("*") if p.is_file() and codes[l] in p.name and split in p.as_posix().split("/")
                     and p.suffix in (".parquet", ".jsonl", ".json")]
            cands = [p for p in cands if (split == "dev") == ("devtest" not in p.as_posix())]
            if not cands:
                raise RuntimeError(f"flores_plus: không thấy tệp cho {l}/{split}; xem cấu trúc {tmp}")
            p = cands[0]
            df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_json(p, lines=p.suffix == ".jsonl")
            if "id" in df.columns:
                df = df.sort_values("id")
            dest = out / split / f"{cfg['languages'][l]['flores']}.{split}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("\n".join(str(t).replace("\n", " ") for t in df["text"]) + "\n", encoding="utf-8")


def fetch_flores(args, cfg) -> None:
    out = RAW / "flores"
    langs = _langs(args, cfg)
    if args.flores_source == "flores_plus":
        fetch_flores_plus(args, cfg, out, langs)
        record_raw("flores", out, cfg["sources"]["flores_plus_repo"], note="flores_plus; [XM]")
        return
    tgz = out / "flores200_dataset.tar.gz"
    last = None
    for url in cfg["sources"]["flores_urls"]:
        try:
            download(url, tgz, force=args.force)
            break
        except Exception as e:  # noqa: BLE001
            last = e
            log.warning("không tải được %s: %s", url, e)
    else:
        raise RuntimeError(f"Không tải được FLORES-200 ({last}). Thử --flores-source flores_plus, "
                           "hoặc tự đặt tệp vào raw/flores/{dev,devtest}/<mã>.<split>")
    n = extract_flores_tar(tgz, out, [cfg["languages"][l]["flores"] for l in langs])
    log.info("FLORES: đã rút %d tệp", n)
    record_raw("flores", out, cfg["sources"]["flores_urls"], archive_sha256=sha256_file(tgz),
               note="FLORES-200 gốc (dev 997 câu, devtest 1012 câu)")


# ------------------------------------------------------------------ NTREX
def ntrex_filename(code: str) -> str:
    return "newstest2019-src.eng.txt" if code == "eng" else f"newstest2019-ref.{code}.txt"


def fetch_ntrex(args, cfg) -> None:
    base = cfg["sources"]["ntrex_base"]
    out = RAW / "ntrex"
    for l in _langs(args, cfg):
        fn = ntrex_filename(cfg["languages"][l]["ntrex"])
        download(f"{base}/NTREX-128/{fn}", out / "NTREX-128" / fn, force=args.force)
    for fn in ("LANGUAGES.tsv", "DOCUMENT_IDS.tsv"):
        download(f"{base}/{fn}", out / fn, force=args.force)
    record_raw("ntrex", out, base)


# ------------------------------------------------------------------ UD
def extract_ud_zip(zpath: Path, dest: Path) -> List[str]:
    dest.mkdir(parents=True, exist_ok=True)
    names = []
    split_re = re.compile(r"-ud-(train|dev|test)\.conllu$")
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():
            parts = Path(n).parts
            if len(parts) != 2:  # chỉ lấy tệp ở gốc repo; bỏ not-to-release/, thư mục nguồn...
                continue
            base = parts[1]
            if split_re.search(base) or base in ("README.md", "LICENSE.txt", "stats.xml"):
                (dest / base).write_bytes(z.read(n))
                names.append(base)
    return names


def fetch_ud(args, cfg) -> None:
    ref = cfg["sources"]["ud_ref"]
    out = RAW / "ud"
    archives: Dict[str, str] = {}
    for l in _langs(args, cfg):
        repo = cfg["languages"][l].get("ud")
        if not repo:
            log.warning("UD: %s không có treebank, bỏ qua", l)
            continue
        z = out / f"{repo}-{ref}.zip"
        download(f"https://codeload.github.com/UniversalDependencies/{repo}/zip/refs/tags/{ref}", z, force=args.force)
        names = extract_ud_zip(z, out / repo)
        if not any(n.endswith(".conllu") for n in names):
            raise RuntimeError(f"{repo}: zip không có tệp *-ud-(train|dev|test).conllu ở gốc")
        archives[repo] = sha256_file(z)
        z.unlink()
    record_raw("ud", out, "https://github.com/UniversalDependencies", revision=ref, archive_sha256=archives)


# ------------------------------------------------------------------ XNLI / Belebele / OPUS (qua HF snapshot)
def fetch_xnli(args, cfg) -> None:
    from huggingface_hub import snapshot_download

    out = RAW / "xnli"
    repo = cfg["sources"]["xnli_repo"]
    for l in _langs(args, cfg):
        iso = cfg["languages"][l].get("xnli")
        if not iso:
            log.warning("XNLI: %s không có, bỏ qua", l)
            continue
        pats = [f"{iso}/*"] if l == "eng" else [f"{iso}/validation-*", f"{iso}/test-*"]
        snapshot_download(repo, repo_type="dataset", local_dir=out, allow_patterns=pats, token=_hf_token())
        if not list((out / iso).glob("*.parquet")):
            raise RuntimeError(f"XNLI/{iso}: không thấy parquet; cấu trúc repo đã đổi? Xem {out}")
    record_raw("xnli", out, repo)


def fetch_belebele(args, cfg) -> None:
    out = RAW / "belebele"
    langs = _langs(args, cfg)
    codes = {cfg["languages"][l]["flores"] for l in langs}
    if args.belebele_source == "hf":
        from huggingface_hub import snapshot_download

        snapshot_download(cfg["sources"]["belebele_repo"], repo_type="dataset", local_dir=out,
                          allow_patterns=[f"*{c}*" for c in codes], token=_hf_token())
        record_raw("belebele", out, cfg["sources"]["belebele_repo"])
        return
    z = out / "Belebele.zip"
    download(cfg["sources"]["belebele_zip"], z, force=args.force)
    found = set()
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            b = Path(n).name
            if b.endswith(".jsonl") and b[:-6] in codes:
                (out / "data").mkdir(exist_ok=True)
                (out / "data" / b).write_bytes(zf.read(n))
                found.add(b[:-6])
    if codes - found:
        raise RuntimeError(f"Belebele: thiếu {sorted(codes - found)}")
    sha = sha256_file(z)
    z.unlink()
    record_raw("belebele", out, cfg["sources"]["belebele_zip"], archive_sha256=sha)


def opus_pair(cfg, lang: str) -> str:
    return "-".join(sorted(["en", cfg["languages"][lang]["opus"]]))


def fetch_opus100(args, cfg) -> None:
    from huggingface_hub import snapshot_download

    out = RAW / "opus100"
    repo = cfg["sources"]["opus100_repo"]
    for l in _langs(args, cfg, cfg["opus_languages"]):
        pair = opus_pair(cfg, l)
        snapshot_download(repo, repo_type="dataset", local_dir=out, allow_patterns=[f"{pair}/train-*"], token=_hf_token())
        if not list((out / pair).glob("train-*")):
            raise RuntimeError(f"OPUS-100: không thấy cặp {pair} (cặp có thể không tồn tại hoặc cấu trúc đổi)")
    record_raw("opus100", out, repo)


# ------------------------------------------------------------------ MUSE / GlotLID / WALS
def fetch_muse(args, cfg) -> None:
    import requests

    base = cfg["sources"]["muse_base"]
    out = RAW / "muse" / "dictionaries"
    missing = []
    for l in _langs(args, cfg):
        iso = cfg["languages"][l].get("muse")
        if not iso or l == "eng":
            if l != "eng":
                missing.append(f"{l}: MUSE không có")
            continue
        for a, b in (("en", iso), (iso, "en")):
            try:
                download(f"{base}/{a}-{b}.txt", out / f"{a}-{b}.txt", force=args.force)
            except requests.HTTPError as e:
                missing.append(f"{a}-{b}: {e}")
                log.warning("MUSE %s-%s: %s", a, b, e)
    record_raw("muse", out.parent, base, missing=missing,
               note="MUSE chỉ có cặp en-xx / xx-en (trừ nhóm châu Âu)")


def fetch_glotlid(args, cfg) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    repo = cfg["sources"]["glotlid_repo"]
    api = HfApi()
    files = [f for f in api.list_repo_files(repo, token=_hf_token()) if f.endswith(".bin")]
    if not files:
        raise RuntimeError(f"{repo}: không có tệp .bin")
    pick = next((f for f in ("model_v3.bin", "model.bin") if f in files), sorted(files)[-1])
    hf_hub_download(repo, pick, local_dir=RAW / "glotlid", token=_hf_token())
    record_raw("glotlid", RAW / "glotlid", repo, revision=api.model_info(repo, token=_hf_token()).sha, picked=pick)


def fetch_wals(args, cfg) -> None:
    ref = cfg["sources"]["wals_ref"]
    out = RAW / "wals"
    z = out / f"wals-{ref}.zip"
    download(f"https://codeload.github.com/cldf-datasets/wals/zip/refs/tags/{ref}", z, force=args.force)
    n = 0
    with zipfile.ZipFile(z) as zf:
        for name in zf.namelist():
            p = Path(name)
            if p.parent.name == "cldf" and p.name in ("languages.csv", "parameters.csv", "values.csv", "codes.csv"):
                (out / "cldf").mkdir(exist_ok=True)
                (out / "cldf" / p.name).write_bytes(zf.read(name))
                n += 1
    if n != 4:
        raise RuntimeError("WALS: thiếu tệp CLDF")
    sha = sha256_file(z)
    z.unlink()
    record_raw("wals", out, "https://github.com/cldf-datasets/wals", revision=ref, archive_sha256=sha)


# ------------------------------------------------------------------ FineWeb-2 / FineWeb (streaming, giới hạn byte)
def fetch_fineweb2(args, cfg) -> None:
    from datasets import load_dataset

    if args.bytes_per_lang:
        default = list(cfg["configs"]["L8"]) + [l for l in cfg["wordfreq_languages"] if l not in cfg["configs"]["L8"]]
        langs = _langs(args, cfg, default)
        budgets = {l: int(args.bytes_per_lang) for l in langs}
    else:
        prem = load_premium()
        budgets = per_language_budget(prem, cfg, args.scale)
        langs = _langs(args, cfg, list(budgets))
    margin = float(cfg["mono"]["raw_margin"])
    src = cfg["sources"]
    for l in langs:
        if l not in budgets:
            raise SystemExit(f"{l} không có trong kế hoạch ngân sách (configs/wordfreq_languages)")
        target = int(budgets[l] * margin)
        outdir = RAW / ("fineweb" if l == "eng" else "fineweb2") / l
        done = outdir / "_DONE.json"
        if done.exists() and not args.force:
            log.info("%s: đã xong, bỏ qua", l)
            continue
        if outdir.exists():
            shutil.rmtree(outdir)
        if l == "eng":
            repo, name = src["fineweb_repo"], src["fineweb_en_config"]
        else:
            repo, name = src["fineweb2_repo"], cfg["languages"][l]["fineweb2"]
        log.info("%s: stream %s/%s, mục tiêu %.1f MB", l, repo, name, target / 1e6)
        ds = load_dataset(repo, name=name, split="train", streaming=True, token=_hf_token())
        ds = ds.shuffle(seed=cfg["seed"], buffer_size=10_000)
        w = ShardWriter(outdir, "part", 200_000_000, gz=True)
        try:
            from tqdm import tqdm

            bar = tqdm(total=target, unit="B", unit_scale=True, desc=l)
        except ImportError:
            bar = None
        got = docs = 0
        for r in ds:
            t = r.get("text") or ""
            b = len(t.encode("utf-8"))
            w.write({"id": r.get("id"), "text": t, "url": r.get("url"), "dump": r.get("dump")})
            got += b
            docs += 1
            if bar is not None:
                bar.update(b)
            if got >= target:
                break
        w.close()
        if bar is not None:
            bar.close()
        done.write_text(json.dumps({"repo": repo, "config": name, "bytes": got, "docs": docs, "target": target,
                                    "finished_at": utcnow(), "complete": got >= target}, indent=2), encoding="utf-8")
        if got < target:
            log.warning("%s: nguồn cạn trước mục tiêu (%d < %d byte)", l, got, target)
    for sub in ("fineweb", "fineweb2"):
        if (RAW / sub).exists() and any((RAW / sub).glob("*/_DONE.json")):
            record_raw(sub, RAW / sub, src["fineweb_repo"] if sub == "fineweb" else src["fineweb2_repo"],
                       note="mẫu streaming có xáo trộn cục bộ (seed=%s), không phải toàn bộ corpus" % cfg["seed"])


# ------------------------------------------------------------------ verify
def verify(args, cfg) -> int:
    """Kiểm tra các nguồn [XM] mà không tải dữ liệu. Trả về số lỗi."""
    results: Dict[str, dict] = {}
    s = cfg["sources"]

    def rec(name, ok, detail):
        results[name] = {"ok": bool(ok), "detail": detail}
        print(f"  [{'OK ' if ok else 'LỖI'}] {name}: {detail}")

    targets = args.targets
    langs = _langs(args, cfg)
    print("== Kiểm tra nguồn (không tải dữ liệu) ==")
    if "flores" in targets:
        ok = [url_ok(u) for u in s["flores_urls"]]
        rec("flores: tarball", any(o for o, _ in ok), "; ".join(f"{u.split('/')[2]}={d}" for u, (_, d) in zip(s["flores_urls"], ok)))
    if "ntrex" in targets:
        for l in langs:
            fn = ntrex_filename(cfg["languages"][l]["ntrex"])
            rec(f"ntrex: {l}", *url_ok(f"{s['ntrex_base']}/NTREX-128/{fn}"))
    if "ud" in targets:
        for l in langs:
            repo = cfg["languages"][l].get("ud")
            if repo:
                rec(f"ud: {l} {repo}@{s['ud_ref']}", *url_ok(f"https://codeload.github.com/UniversalDependencies/{repo}/zip/refs/tags/{s['ud_ref']}"))
    if "wals" in targets:
        rec("wals", *url_ok(f"https://codeload.github.com/cldf-datasets/wals/zip/refs/tags/{s['wals_ref']}"))
    if "belebele" in targets:
        rec("belebele zip", *url_ok(s["belebele_zip"]))
    if "muse" in targets:
        for l in langs:
            iso = cfg["languages"][l].get("muse")
            if iso and l != "eng":
                rec(f"muse: en-{iso}", *url_ok(f"{s['muse_base']}/en-{iso}.txt"))
                rec(f"muse: {iso}-en", *url_ok(f"{s['muse_base']}/{iso}-en.txt"))
            elif l != "eng":
                rec(f"muse: {l}", False, "MUSE không có ngôn ngữ này (bình thường)")
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        tok = _hf_token()
        if "xnli" in targets:
            files = api.list_repo_files(s["xnli_repo"], repo_type="dataset", token=tok)
            for l in langs:
                iso = cfg["languages"][l].get("xnli")
                if iso:
                    rec(f"xnli: {iso}", any(f.startswith(f"{iso}/") and f.endswith(".parquet") for f in files), "parquet trong repo")
        if "opus100" in targets:
            files = api.list_repo_files(s["opus100_repo"], repo_type="dataset", token=tok)
            for l in cfg["opus_languages"]:
                pair = opus_pair(cfg, l)
                rec(f"opus100: {pair}", any(f.startswith(f"{pair}/train") for f in files), "train trong repo")
        if "fineweb2" in targets:
            from datasets import get_dataset_config_names

            names = set(get_dataset_config_names(s["fineweb2_repo"]))
            for l in cfg["wordfreq_languages"]:
                c = cfg["languages"][l]["fineweb2"]
                if c:
                    rec(f"fineweb2: {l} -> {c}", c in names, "config có trong repo" if c in names else "KHÔNG có config này")
            names_en = set(get_dataset_config_names(s["fineweb_repo"]))
            rec(f"fineweb: {s['fineweb_en_config']}", s["fineweb_en_config"] in names_en, "config tiếng Anh")
        if "glotlid" in targets:
            fl = [f for f in api.list_repo_files(s["glotlid_repo"], token=tok) if f.endswith(".bin")]
            rec("glotlid", bool(fl), ", ".join(fl))
        if "flores" in targets:
            try:
                api.dataset_info(s["flores_plus_repo"], token=tok)
                rec("flores_plus (dự phòng)", True, "repo truy cập được")
            except Exception as e:  # noqa: BLE001
                rec("flores_plus (dự phòng)", False, f"{type(e).__name__} (cần HF_TOKEN + chấp nhận điều khoản?)")
    except ImportError:
        print("  (bỏ qua kiểm tra HF: chưa cài huggingface_hub/datasets)")
    except Exception as e:  # noqa: BLE001
        rec("hugging face", False, f"{type(e).__name__}: {e}")
    update_manifest("verification", "last_run", {"at": utcnow(), "results": results})
    bad = [k for k, v in results.items() if not v["ok"]]
    print(f"\n{len(results) - len(bad)}/{len(results)} OK. (Lỗi 'MUSE không có ngôn ngữ này' là bình thường.)")
    return len(bad)


DISPATCH = {"flores": fetch_flores, "ntrex": fetch_ntrex, "xnli": fetch_xnli, "ud": fetch_ud, "belebele": fetch_belebele,
            "muse": fetch_muse, "glotlid": fetch_glotlid, "wals": fetch_wals, "fineweb2": fetch_fineweb2,
            "opus100": fetch_opus100}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", help="bộ dữ liệu, hoặc small / large / all (mặc định: small)")
    ap.add_argument("--langs", nargs="+", help="giới hạn ngôn ngữ (mã 3 chữ trong config.yaml)")
    ap.add_argument("--skip", nargs="+", default=[], help="bỏ qua các bộ này, vd: --skip glotlid")
    ap.add_argument("--verify", action="store_true", help="chỉ kiểm tra nguồn, không tải")
    ap.add_argument("--force", action="store_true", help="tải lại dù đã có")
    ap.add_argument("--scale", type=float, default=1.0, help="nhân ngân sách byte (fineweb2), vd 0.01 để chạy thử")
    ap.add_argument("--bytes-per-lang", type=int, help="bỏ qua byte_premium, tải đúng ngần này byte mỗi ngôn ngữ")
    ap.add_argument("--flores-source", choices=["official", "flores_plus"], default="official")
    ap.add_argument("--belebele-source", choices=["official", "hf"], default="official")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)
    cfg = load_config()

    t: List[str] = []
    for x in args.targets or (["all"] if args.verify else ["small"]):
        t += SMALL if x == "small" else LARGE if x == "large" else SMALL + LARGE if x == "all" else [x]
    bad = [x for x in t if x not in DISPATCH]
    if bad:
        raise SystemExit(f"Mục tiêu không hợp lệ: {bad}. Hợp lệ: {list(DISPATCH)} small large all")
    args.targets = [x for x in dict.fromkeys(t) if x not in args.skip]

    if args.verify:
        raise SystemExit(1 if verify(args, cfg) else 0)
    for name in args.targets:
        log.info("=== %s ===", name)
        DISPATCH[name](args, cfg)
    log.info("xong. Manifest: %s", "data/manifest.json")


if __name__ == "__main__":
    main()
