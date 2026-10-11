#!/usr/bin/env python3
"""
download_data.py
================
Download a large multilingual corpus (CC-100, 10 widely used languages) into
``data/cc100/``.

Corpus
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

Usage
-----
    python download_data.py                       # 10 languages, 50 MB each
    python download_data.py --max-mb 200          # bigger sample
    python download_data.py --langs en vi de      # subset
    python download_data.py --force               # re-download
"""
from __future__ import annotations

import argparse
import json
import lzma
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


# --------------------------------------------------------------------------- #
# Download
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
# Loader (reusable from other scripts)
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


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Download a CC-100 sample (10 languages).")
    ap.add_argument("--langs", nargs="+", default=DEFAULT_LANGS, help="CC-100 language codes")
    ap.add_argument("--max-mb", type=float, default=50.0,
                    help="MB of decompressed text kept per language (default: 50)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()

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

    print("\nDone. Files are in:", args.out_dir)
    print("Please cite Conneau et al. (2020) and Wenzek et al. (2020); see the docstring for BibTeX.")


if __name__ == "__main__":
    main()