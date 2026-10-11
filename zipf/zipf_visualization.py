#!/usr/bin/env python3
"""
zipf/zipf_visualization.py
==========================
Three side-by-side panels illustrating Zipf's law on the CC-100 sample
downloaded by ``download_data.py``:

  (a) Rank-frequency (log-log):  f(r) = c * r^(-alpha)               [Eq. zipf]
  (b) Normalised distribution:   p(r) = r^(-alpha) / H_{N,alpha}     [Eq. zipfnorm]
      per language, against the ideal alpha = 1 curve
  (c) Imbalance between languages of the multilingual corpus
      (languages ranked by amount of text)

No titles; sub-captions "(a)", "(b)", "(c)" are placed under each panel.
Legend font size 14, sub-caption font size 24, Times New Roman, PDF output.

Usage
-----
    python zipf/zipf_visualization.py
    python zipf/zipf_visualization.py --recount        # ignore the cached counts
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import regex
from matplotlib import font_manager
from matplotlib.ticker import NullFormatter, ScalarFormatter

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "cc100"
OUT_PDF = Path(__file__).resolve().parent / "zipf_law.pdf"

DEFAULT_LANGS = ["en", "de", "fr", "es", "ru", "vi", "id", "tr", "ar", "hi"]
LANG_NAMES = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish",
    "ru": "Russian", "vi": "Vietnamese", "id": "Indonesian", "tr": "Turkish",
    "ar": "Arabic", "hi": "Hindi",
}

LEGEND_FS = 16     # legend font size
SUBFIG_FS = 28      # (a), (b), (c) font size
LABEL_FS = 24       # axis labels
TICK_FS = 16        # tick labels

FIT_RANGE = (10, 10_000)   # ranks used to estimate alpha in panel (a)

# Letters + combining marks (so Hindi / Arabic words are not split at matras).
TOKEN = regex.compile(r"[\p{L}\p{M}]+")


def setup_style() -> None:
    names = {f.name for f in font_manager.fontManager.ttflist}
    serif = ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "STIXGeneral"]
    if "Times New Roman" not in names:
        print("[warn] 'Times New Roman' not found by matplotlib; falling back to a "
              "Times-compatible serif font.")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": serif,
        "mathtext.fontset": "stix",      # Times-like math
        "pdf.fonttype": 42,              # embed TrueType fonts in the PDF
        "axes.linewidth": 1.2,
    })


# --------------------------------------------------------------------------- #
# Counting
# --------------------------------------------------------------------------- #
def count_words(path: str) -> Counter:
    c: Counter = Counter()
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            c.update(TOKEN.findall(line.lower()))
    return c


def load_counts(langs: list[str], recount: bool) -> dict[str, Counter]:
    cache = DATA_DIR / "word_counts.pkl"
    if cache.exists() and not recount:
        with open(cache, "rb") as f:
            cached = pickle.load(f)
        if all(l in cached for l in langs):
            return {l: cached[l] for l in langs}

    paths = []
    for l in langs:
        p = DATA_DIR / f"{l}.txt"
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run download_data.py first.")
        paths.append(str(p))

    print("Counting words ...")
    with ProcessPoolExecutor() as ex:
        counts = dict(zip(langs, ex.map(count_words, paths)))

    with open(cache, "wb") as f:
        pickle.dump(counts, f, protocol=pickle.HIGHEST_PROTOCOL)
    return counts


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def sorted_freqs(c: Counter) -> np.ndarray:
    return np.array(sorted(c.values(), reverse=True), dtype=np.float64)


def log_ranks(n: int, k: int = 600) -> np.ndarray:
    """Log-spaced 0-based rank indices (keeps log-log plots light)."""
    idx = np.unique(np.logspace(0, np.log10(n), k).astype(np.int64)) - 1
    return np.clip(idx, 0, n - 1)


def fit_zipf(freqs: np.ndarray, lo: int, hi: int) -> tuple[float, float]:
    """Least-squares fit of log f = log c - alpha log r on ranks lo..hi."""
    hi = min(hi, len(freqs))
    r = np.arange(lo, hi + 1)
    slope, intercept = np.polyfit(np.log(r), np.log(freqs[lo - 1:hi]), 1)
    return -slope, float(np.exp(intercept))


def style_axis(ax) -> None:
    ax.tick_params(labelsize=TICK_FS, width=1.2)
    ax.grid(True, which="major", alpha=0.25, linewidth=0.8)


def add_subcaption(ax, text: str) -> None:
    """One-line sub-caption, e.g. '(a) Rank-frequency: f(r)=c r^-alpha', size 24."""
    ax.text(0.5, -0.14, text, transform=ax.transAxes,
            ha="center", va="top", fontsize=SUBFIG_FS)


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def panel_a(ax, pooled: np.ndarray) -> float:
    """Rank-frequency of the pooled corpus + fitted Zipf line + alpha = 1."""
    n = len(pooled)
    idx = log_ranks(n)
    r = idx + 1

    alpha, c = fit_zipf(pooled, *FIT_RANGE)

    ax.loglog(r, pooled[idx], color="black", lw=2.8, label="Corpus")
    lo, hi = FIT_RANGE[0], min(FIT_RANGE[1], n)
    r_fit = np.unique(np.logspace(np.log10(lo), np.log10(hi), 100).astype(np.int64))
    ax.loglog(r_fit, c * r_fit ** (-alpha), color="#d62728", ls="--", lw=3.2,
              label=rf"Fit: $\hat{{\alpha}}={alpha:.2f}$")
    ax.loglog(r, pooled[0] * r ** (-1.0), color="#7f7f7f", ls=":", lw=2.4,
              label=r"$\alpha=1$")

    ax.set_xlabel(r"Rank $r$", fontsize=LABEL_FS)
    ax.set_ylabel(r"Frequency $f(r)$", fontsize=LABEL_FS)
    ax.legend(fontsize=LEGEND_FS, loc="lower left", frameon=False)
    style_axis(ax)
    return alpha


def panel_b(ax, per_lang: dict[str, np.ndarray]) -> None:
    """Empirical p(r) for each language vs. the ideal Zipf distribution."""
    cmap = plt.get_cmap("tab10")
    for i, (lang, freqs) in enumerate(per_lang.items()):
        p = freqs / freqs.sum()
        idx = log_ranks(len(p))
        ax.loglog(idx + 1, p[idx], color=cmap(i), lw=1.8, alpha=0.95,
                  label=LANG_NAMES.get(lang, lang))

    n_max = max(len(f) for f in per_lang.values())
    h = np.sum(1.0 / np.arange(1, n_max + 1))             # H_{N,1}
    r = log_ranks(n_max) + 1
    ax.loglog(r, (1.0 / r) / h, color="black", ls="--", lw=2.6, label=r"Zipf, $\alpha=1$")

    ax.set_xlabel(r"Rank $r$", fontsize=LABEL_FS)
    ax.set_ylabel(r"Probability $p(r)$", fontsize=LABEL_FS)
    ax.legend(fontsize=LEGEND_FS, loc="lower left", ncol=2, frameon=False,
              columnspacing=1.0, handlelength=1.6)
    style_axis(ax)


def panel_c(ax, langs: list[str], counts: dict[str, Counter]) -> None:
    """Languages ranked by amount of text (full CC-100 size if available)."""
    meta_path = DATA_DIR / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    sizes = {l: (meta.get(l, {}).get("full_xz_bytes") or 0) / 1e9 for l in langs}
    if all(v > 0 for v in sizes.values()):
        ylabel, legend_label = "Size (GB)", "CC-100 (full)"
    else:
        print("[warn] Full CC-100 sizes missing in meta.json; using sampled tokens instead.")
        sizes = {l: sum(counts[l].values()) / 1e6 for l in langs}
        ylabel, legend_label = "Tokens (M)", "Corpus sample"

    order = sorted(langs, key=lambda l: sizes[l], reverse=True)
    ranks = np.arange(1, len(order) + 1)
    vals = np.array([sizes[l] for l in order])

    # Linear x with language codes as ticks (no overlapping labels), log y.
    ax.semilogy(ranks, vals, "o-", color="black", lw=2.2, ms=9, label=legend_label)
    ax.semilogy(ranks, vals[0] * ranks ** (-1.0), color="#7f7f7f", ls=":", lw=2.4,
                label=r"$\alpha=1$")

    ax.set_xticks(ranks)
    ax.set_xticklabels([l.upper() for l in order])
    ax.set_xlim(0.6, len(order) + 0.4)
    ymin, ymax = vals.min(), vals.max()
    ax.set_ylim(ymin / 1.6, ymax * 1.6)

    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200, 500) if ymin / 1.6 <= t <= ymax * 1.6]
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.set_xlabel("Language", fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.legend(fontsize=LEGEND_FS, loc="upper right", frameon=False)
    style_axis(ax)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=DEFAULT_LANGS)
    ap.add_argument("--out", type=Path, default=OUT_PDF)
    ap.add_argument("--recount", action="store_true", help="ignore cached word counts")
    args = ap.parse_args()

    setup_style()
    counts = load_counts(args.langs, args.recount)

    pooled_counter: Counter = Counter()
    for c in counts.values():
        pooled_counter.update(c)
    pooled = sorted_freqs(pooled_counter)
    per_lang = {l: sorted_freqs(c) for l, c in counts.items()}

    for l, c in counts.items():
        print(f"{l}: {sum(c.values()):>12,} tokens | {len(c):>9,} types")
    print(f"pooled: {int(pooled.sum()):,} tokens | {len(pooled):,} types")

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.4))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.98, bottom=0.21, wspace=0.27)

    alpha = panel_a(axes[0], pooled)
    panel_b(axes[1], per_lang)
    panel_c(axes[2], args.langs, counts)

    captions = [
        r"(a) Rank$-$frequency: $f(r)=c\,r^{-\alpha}$",
        r"(b) $p(r)=r^{-\alpha}/H_{N,\alpha}$",
        "(c) Language imbalance",
    ]
    for ax, text in zip(axes, captions):
        add_subcaption(ax, text)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Fitted alpha (pooled, ranks {FIT_RANGE[0]}-{FIT_RANGE[1]}): {alpha:.3f}")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()