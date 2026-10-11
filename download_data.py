#!/usr/bin/env python3
"""
download_data.py
================
Download the datasets used in this repo into ``data/``:

  1. CC-100 sample (10 widely used languages)  -> ``data/cc100/<lang>.txt``
  2. FLORES-200 (parallel, 200+ languages)     -> ``data/flores.json``

CC-100
------
CC-100 is a monolingual web corpus covering 100+ languages, built with the
CCNet pipeline on Common Crawl and used to train XLM-R. Files are served as
``https://data.statmt.org/cc-100/<lang>.txt.xz`` (one sentence per line,
documents separated by an empty line).

Full files are huge (English alone is tens of GB), so this script *streams*
the xz archive and keeps only the first ``--max-mb`` MB of decompressed text
per language (cut at a line boundary). The full archive size is read from the
HTTP ``Content-Length`` header and stored in ``data/cc100/meta.json``; it is
used by ``zipf/zipf_visualization.py`` to show the language imbalance.

FLORES-200
----------
The official release ``flores200_dataset.tar.gz`` (NLLB Team, ~25 MB) holds one
text file per language and split (``dev``: 997 sentences, ``devtest``: 1012
sentences; line *i* of every file is the translation of the same English
sentence). It is downloaded, converted to a single JSON file and the archive is
deleted again (use ``--keep-raw`` to keep it).

Output format of ``data/flores.json`` (what ``moe_rounting_js.py`` reads): a
list with one record per parallel sentence, keyed by FLORES-200 language code

    [
      {"id": "dev_0", "split": "dev",
       "eng_Latn": "On Monday, scientists from ...",
       "vie_Latn": "Vào thứ Hai, các nhà khoa học ...",
       "zho_Hans": "...", ...},
      ...
    ]

``id`` and ``split`` are metadata; every key matching ``xxx_Xxxx`` is a language.

Please cite
-----------
@inproceedings{conneau2020unsupervised,
  title     = {Unsupervised Cross-lingual Representation Learning at Scale},
  author    = {Conneau, Alexis and Khandelwal, Kartikay and Goyal, Naman and
               Chaudhary, Vishrav and Wenzek, Guillaume and Guzm{\\'a}n, Francisco
               and Grave, Edouard and Ott, Myle and Zettlemoyer, Luke and
               Stoyanov, Veselin},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for
               Computational Linguistics (ACL)},
  pages     = {8440--8451},
  year      = {2020}
}

@inproceedings{wenzek2020ccnet,
  title     = {{CCN}et: Extracting High Quality Monolingual Datasets from Web
               Crawl Data},
  author    = {Wenzek, Guillaume and Lachaux, Marie-Anne and Conneau, Alexis
               and Chaudhary, Vishrav and Guzm{\\'a}n, Francisco and Joulin,
               Armand and Grave, Edouard},
  booktitle = {Proceedings of the 12th Language Resources and Evaluation
               Conference (LREC)},
  pages     = {4003--4012},
  year      = {2020}
}

@article{nllb2022,
  title   = {No Language Left Behind: Scaling Human-Centered Machine Translation},
  author  = {{NLLB Team} and Costa-juss{\\`a}, Marta R. and others},
  journal = {arXiv preprint arXiv:2207.04672},
  year    = {2022}
}

Usage
-----
    python download_data.py                          # everything: CC-100 + FLORES-200 (default)
    python download_data.py --dataset cc100          # CC-100 only
    python download_data.py --dataset flores         # FLORES-200 only -> data/flores.json

    python download_data.py --max-mb 200             # bigger CC-100 sample
    python download_data.py --langs en vi de         # CC-100 subset
    python download_data.py --dataset flores --flores-splits devtest
    python download_data.py --dataset flores --flores-archive ~/Downloads/flores200_dataset.tar.gz
    python download_data.py --dataset flores --force # rebuild flores.json
"""
from __future__ import annotations

import argparse
import json
import lzma
import re
import tarfile
import time
from pathlib import Path
from typing import Iterator

import requests
from tqdm import tqdm

BASE_URL = "https://data.statmt.org/cc-100"

# 10 widely used languages with whitespace-delimited words (needed for a
# word-level Zipf analysis): en, de, fr, es, ru, vi, id, tr, ar, hi.
DEFAULT_LANGS = ["en", "de", "fr", "es", "ru", "vi", "id", "tr", "ar", "hi"]

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "data" / "cc100"

# FLORES-200 -----------------------------------------------------------------
FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
DEFAULT_FLORES_OUT = ROOT / "data" / "flores.json"
FLORES_SPLITS = ["dev", "devtest"]
ENGLISH = "eng_Latn"
# e.g. flores200_dataset/devtest/vie_Latn.devtest
FLORES_MEMBER_RE = re.compile(r"(?:^|/)(dev|devtest)/([a-z]{3}_[A-Z][a-z]{3})\.(dev|devtest)$")


# --------------------------------------------------------------------------- #
# CC-100 download
# --------------------------------------------------------------------------- #
def _stream_one(lang: str, out_dir: Path, limit: int) -> dict:
    """Stream ``<lang>.txt.xz`` and write the first ``limit`` bytes of text."""
    url = f"{BASE_URL}/{lang}.txt.xz"
    out_path = out_dir / f"{lang}.txt"
    tmp_path = out_dir / f"{lang}.txt.part"

    decomp = lzma.LZMADecompressor()
    written = 0

    with requests.get(url, stream=True, timeout=(15, 120)) as resp:
        resp.raise_for_status()
        full_xz_bytes = int(resp.headers.get("Content-Length", 0)) or None

        with open(tmp_path, "wb") as f, tqdm(
            total=limit, unit="B", unit_scale=True, desc=f"[{lang}]"
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                data = decomp.decompress(chunk)

                if written + len(data) >= limit:
                    data = data[: limit - written]
                    cut = data.rfind(b"\n")          # keep whole lines only
                    if cut != -1:
                        data = data[: cut + 1]
                    f.write(data)
                    written += len(data)
                    bar.update(len(data))
                    break

                f.write(data)
                written += len(data)
                bar.update(len(data))
                if decomp.eof:
                    break

    tmp_path.replace(out_path)
    return {"sampled_bytes": written, "full_xz_bytes": full_xz_bytes}


def download_language(lang: str, out_dir: Path, max_mb: float, retries: int = 3) -> dict:
    limit = int(max_mb * 1024 * 1024)
    for attempt in range(1, retries + 1):
        try:
            return _stream_one(lang, out_dir, limit)
        except (requests.RequestException, lzma.LZMAError, EOFError) as exc:
            print(f"[{lang}] attempt {attempt}/{retries} failed: {exc}")
            time.sleep(3 * attempt)
    raise RuntimeError(f"Could not download '{lang}' after {retries} attempts.")


# --------------------------------------------------------------------------- #
# FLORES-200 download + processing
# --------------------------------------------------------------------------- #
def _download_file(url: str, dest: Path, retries: int = 3) -> None:
    """Download ``url`` to ``dest`` (via a ``.part`` file), with a progress bar."""
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=(15, 120)) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length", 0)) or None
                with open(tmp, "wb") as f, tqdm(
                    total=total, unit="B", unit_scale=True, desc="[flores]"
                ) as bar:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
            tmp.replace(dest)
            return
        except requests.RequestException as exc:
            print(f"[flores] attempt {attempt}/{retries} failed: {exc}")
            time.sleep(3 * attempt)
    raise RuntimeError(
        f"Could not download {url} after {retries} attempts. "
        "Download it manually and pass it with --flores-archive."
    )


def build_flores_records(archive: Path, splits: list[str]) -> list[dict]:
    """Read ``flores200_dataset.tar.gz`` -> list of parallel records (see module docstring)."""
    per_split: dict[str, dict[str, list[str]]] = {s: {} for s in splits}

    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            m = FLORES_MEMBER_RE.search(member.name)
            if not m:
                continue
            split, lang, ext = m.group(1), m.group(2), m.group(3)
            if split != ext or split not in per_split:
                continue
            # split on "\n" only (str.splitlines would also cut on U+2028 etc.)
            lines = tar.extractfile(member).read().decode("utf-8").split("\n")
            if lines and lines[-1] == "":
                lines.pop()
            per_split[split][lang] = lines

    records: list[dict] = []
    for split in splits:
        by_lang = per_split[split]
        if not by_lang:
            raise RuntimeError(f"No '{split}' files found in {archive}.")
        if ENGLISH not in by_lang:
            raise RuntimeError(f"'{ENGLISH}' missing from the '{split}' split.")

        n = len(by_lang[ENGLISH])
        bad = {l: len(v) for l, v in by_lang.items() if len(v) != n}
        if bad:
            raise RuntimeError(f"[{split}] languages not aligned with English ({n} lines): {bad}")

        langs = [ENGLISH] + sorted(l for l in by_lang if l != ENGLISH)
        print(f"[flores] {split}: {n} sentences x {len(langs)} languages")
        for i in range(n):
            rec = {"id": f"{split}_{i}", "split": split}
            for lang in langs:
                rec[lang] = by_lang[lang][i]
            records.append(rec)
    return records


def prepare_flores(
    out_path: Path,
    splits: list[str],
    url: str = FLORES_URL,
    archive: Path | None = None,
    keep_raw: bool = False,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    downloaded = False
    if archive is None:
        archive = out_path.parent / "flores200_dataset.tar.gz"
        if not archive.exists():
            _download_file(url, archive)
            downloaded = True
    elif not archive.exists():
        raise FileNotFoundError(archive)

    records = build_flores_records(archive, splits)

    tmp = out_path.with_name(out_path.name + ".part")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    tmp.replace(out_path)
    print(f"[flores] wrote {len(records)} records -> {out_path} "
          f"({out_path.stat().st_size / 1e6:.1f} MB)")

    if downloaded and not keep_raw:
        archive.unlink()


# --------------------------------------------------------------------------- #
# Loaders (reusable from other scripts)
# --------------------------------------------------------------------------- #
def iter_corpus(
    langs: list[str] | None = None, data_dir: Path | str = DEFAULT_OUT
) -> Iterator[tuple[str, str]]:
    """Yield ``(lang, line)`` for every non-empty line of the downloaded corpus."""
    data_dir = Path(data_dir)
    for lang in langs or DEFAULT_LANGS:
        path = data_dir / f"{lang}.txt"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run download_data.py first.")
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield lang, line


def load_flores(path: Path | str = DEFAULT_FLORES_OUT) -> list[dict]:
    """Load ``data/flores.json`` (list of parallel records)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python download_data.py --dataset flores")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def download_cc100(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.out_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    for lang in args.langs:
        target = args.out_dir / f"{lang}.txt"
        if target.exists() and not args.force:
            print(f"[{lang}] already exists, skipping (use --force to re-download).")
            continue
        meta[lang] = download_language(lang, args.out_dir, args.max_mb)
        meta_path.write_text(json.dumps(meta, indent=2))

    print("\nCC-100 done. Files are in:", args.out_dir)
    print("Please cite Conneau et al. (2020) and Wenzek et al. (2020); see the docstring for BibTeX.")


def download_flores(args: argparse.Namespace) -> None:
    if args.flores_out.exists() and not args.force:
        print(f"[flores] {args.flores_out} already exists, skipping (use --force to rebuild).")
        return
    prepare_flores(
        args.flores_out,
        args.flores_splits,
        url=args.flores_url,
        archive=args.flores_archive,
        keep_raw=args.keep_raw,
    )
    print("\nFLORES-200 done. Please cite NLLB Team et al. (2022); see the docstring for BibTeX.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download CC-100 (10 languages) and/or FLORES-200.")
    ap.add_argument("--dataset", choices=["cc100", "flores", "all"], default="all",
                    help="what to download (default: all = CC-100 + FLORES-200)")
    ap.add_argument("--force", action="store_true", help="re-download / rebuild existing files")

    g1 = ap.add_argument_group("CC-100")
    g1.add_argument("--langs", nargs="+", default=DEFAULT_LANGS, help="CC-100 language codes")
    g1.add_argument("--max-mb", type=float, default=50.0,
                    help="MB of decompressed text kept per language (default: 50)")
    g1.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)

    g2 = ap.add_argument_group("FLORES-200")
    g2.add_argument("--flores-out", type=Path, default=DEFAULT_FLORES_OUT,
                    help="output json (default: data/flores.json)")
    g2.add_argument("--flores-splits", nargs="+", choices=FLORES_SPLITS, default=FLORES_SPLITS,
                    help="FLORES splits to merge (default: dev devtest)")
    g2.add_argument("--flores-url", default=FLORES_URL, help="archive URL")
    g2.add_argument("--flores-archive", type=Path, default=None,
                    help="use an already downloaded flores200_dataset.tar.gz instead of downloading")
    g2.add_argument("--keep-raw", action="store_true", help="keep the downloaded .tar.gz")
    args = ap.parse_args()

    if args.dataset in ("cc100", "all"):
        download_cc100(args)
    if args.dataset in ("flores", "all"):
        download_flores(args)


if __name__ == "__main__":
    main()