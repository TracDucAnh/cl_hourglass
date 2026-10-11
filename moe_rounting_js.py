#!/usr/bin/env python3
"""
Routing-divergence ("U-shape") figure for Qwen1.5-MoE-A2.7B over all FLORES-200 languages.

For every FLORES language and every MoE layer we measure how differently the router
treats a sentence compared with its (parallel) English translation, following
Bandarkar et al., "Multilingual Routing in Mixture-of-Experts" (ICLR 2026), Sec. 4.3
and Appendix A.4:

  * q_i^(lang,l)  = mean over the tokens of sequence i of the post-softmax routing
                    weights of layer l (full E-dim distribution, Eq. 1)
  * D_H-JS(q_en || q_lang) = JSD(q_en, q_lang) / (log E - (H(q_en) + H(q_lang)) / 2)
                    (entropy-normalized Jensen-Shannon divergence, Eq. 6-8, natural log)
  * Div(lang, l)  = mean over sequences of D_H-JS (Eq. 2)

Figure (single PDF, two panels, no title, Times New Roman):
  (a) JS divergence from English of every FLORES language (one thin line each),
      with the 10 most widely spoken languages highlighted in the legend
  (b) mean +- 1 std of the JS divergence across all languages

Usage (run from the repo root, CL_Hourglass/)
---------------------------------------------
    python moe_rounting_js.py \\
        --model_name_or_path Qwen/Qwen1.5-MoE-A2.7B \\
        --flores_path data/flores.json \\
        --num_samples 200 --batch_size 32 \\
        --output_dir results/routing_divergence

    # shade the layers you intervene in (e.g. 8-16)
    ... --mark_layers 8 16

    # re-draw the figure from the cached numbers (no model / GPU needed)
    ... --plot_only

Outputs (in --output_dir)
-------------------------
    routing_divergence.json   per-language x per-layer divergence (also the resume checkpoint)
    routing_divergence.csv    same numbers as a table
    routing_divergence.pdf    the figure

Resume: languages already present in routing_divergence.json (same settings) are skipped.
Use --overwrite to start from scratch.
"""

import argparse
import colorsys
import gc
import json
import os
import random
import re
import sys
import tempfile

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ENGLISH = "eng_Latn"
LANG_KEY_RE = re.compile(r"^[a-z]{3}_[A-Z][a-z]{3}$")  # FLORES-200 codes, e.g. vie_Latn
EPS = 1e-12

# ----------------------------- figure font sizes ----------------------------- #
LEGEND_FS = 16     # legend font size
SUBFIG_FS = 28     # (a), (b) font size
LABEL_FS = 24      # axis labels
TICK_FS = 16       # tick labels

# Ten most widely spoken languages other than English (total speakers).
DEFAULT_LEGEND_LANGS = [
    "zho_Hans", "hin_Deva", "spa_Latn", "fra_Latn", "arb_Arab",
    "ben_Beng", "por_Latn", "rus_Cyrl", "urd_Arab", "ind_Latn",
]
LANG_NAMES = {
    "zho_Hans": "Chinese", "hin_Deva": "Hindi", "spa_Latn": "Spanish",
    "fra_Latn": "French", "arb_Arab": "Arabic", "ben_Beng": "Bengali",
    "por_Latn": "Portuguese", "rus_Cyrl": "Russian", "urd_Arab": "Urdu",
    "ind_Latn": "Indonesian",
}


# ----------------------------------------------------------------------------- #
# Divergence maths (numpy)
# ----------------------------------------------------------------------------- #
def _entropy(p):
    return -(p * np.log(p + EPS)).sum(axis=-1)


def entropy_normalized_jsd(q1, q2):
    """q1, q2: (..., E) probability vectors -> (...) entropy-normalized JS divergence in [0, 1]."""
    E = q1.shape[-1]
    m = 0.5 * (q1 + q2)
    log_m = np.log(m + EPS)
    kl1 = (q1 * (np.log(q1 + EPS) - log_m)).sum(axis=-1)
    kl2 = (q2 * (np.log(q2 + EPS) - log_m)).sum(axis=-1)
    jsd = 0.5 * (kl1 + kl2)
    norm = np.log(E) - 0.5 * (_entropy(q1) + _entropy(q2))  # F = log E - H_avg
    return np.where(norm > 1e-8, jsd / np.maximum(norm, 1e-8), np.nan)


# ----------------------------------------------------------------------------- #
# Router recording
# ----------------------------------------------------------------------------- #
def is_oom_error(err):
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(err, oom_cls):
        return True
    return isinstance(err, RuntimeError) and "out of memory" in str(err).lower()


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class RoutingRecorder:
    """Forward hooks on every MoE router (`mlp.gate`, nn.Linear) that store, per sequence,
    the token-averaged post-softmax routing distribution q in [0,1]^E for each layer."""

    def __init__(self, model):
        self.backbone = model.model
        self.gates = {}
        for li, layer in enumerate(self.backbone.layers):
            gate = getattr(getattr(layer, "mlp", None), "gate", None)
            if gate is None:
                continue  # dense MLP layer
            if not isinstance(gate, torch.nn.Linear):
                raise RuntimeError(
                    f"Layer {li}: router is {type(gate).__name__}, expected nn.Linear "
                    "(use transformers>=4.40,<5)."
                )
            self.gates[li] = gate
        if not self.gates:
            raise RuntimeError("No MoE router (`layer.mlp.gate`) found.")
        self.layer_ids = sorted(self.gates)
        self.num_experts = self.gates[self.layer_ids[0]].out_features
        self._mask, self._q = None, {}
        for li, g in self.gates.items():
            g.register_forward_hook(self._make_hook(li))

    def _make_hook(self, li):
        def hook(module, inputs, output):
            if self._mask is None:
                return None
            B, T = self._mask.shape
            mask = self._mask.to(output.device)
            p = torch.softmax(output.float(), dim=-1).reshape(B, T, -1)  # (B, T, E)
            q = (p * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            self._q[li] = q.cpu()
            return None

        return hook

    @torch.no_grad()
    def _forward(self, tokenizer, texts, device, max_length):
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length, add_special_tokens=False)
        self._mask, self._q = enc["attention_mask"].bool(), {}
        try:  # backbone only: skips the 150k-vocab LM head, all routers still run
            self.backbone(input_ids=enc["input_ids"].to(device),
                          attention_mask=enc["attention_mask"].to(device), use_cache=False)
        finally:
            self._mask = None
        return torch.stack([self._q[li] for li in self.layer_ids], dim=1)  # (B, L, E)

    def route_batch(self, tokenizer, texts, device, max_length):
        """_forward with recursive batch halving on OOM (retry only after the except block ended)."""
        oom = False
        try:
            return self._forward(tokenizer, texts, device, max_length)
        except RuntimeError as e:
            if not is_oom_error(e):
                raise
            oom = True
        clear_memory()
        if oom and len(texts) == 1:
            raise RuntimeError("Out of memory even with a single sequence; lower --max_length.")
        mid = len(texts) // 2
        left = self.route_batch(tokenizer, texts[:mid], device, max_length)
        right = self.route_batch(tokenizer, texts[mid:], device, max_length)
        return torch.cat([left, right], dim=0)

    def profiles(self, tokenizer, texts, device, batch_size, max_length):
        """-> float32 array (len(texts), L, E) of per-sequence mean routing weights."""
        out = np.zeros((len(texts), len(self.layer_ids), self.num_experts), dtype=np.float32)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar lengths together
        for s in range(0, len(order), batch_size):
            idx = order[s : s + batch_size]
            out[idx] = self.route_batch(tokenizer, [texts[i] for i in idx], device, max_length).numpy()
        return out


# ----------------------------------------------------------------------------- #
# Data / cache helpers
# ----------------------------------------------------------------------------- #
def load_flores(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("data", "examples", "rows"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"Unrecognized FLORES json structure at {path}")
    return data


def atomic_write_json(path, obj):
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_div_", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def lang_label(code):
    return LANG_NAMES.get(code, code)


# ----------------------------------------------------------------------------- #
# Plot
# ----------------------------------------------------------------------------- #
def distinct_colors(n):
    """n visually different colours: golden-ratio hues x cycling saturation/brightness."""
    levels = [(0.85, 0.90), (0.70, 0.75), (0.95, 0.65), (0.55, 0.95), (0.80, 0.55)]
    cols, seen = [], set()
    i = 0
    while len(cols) < n:
        h = (i * 0.6180339887498949) % 1.0
        s, v = levels[i % len(levels)]
        rgb = colorsys.hsv_to_rgb(h, s, v)
        hexc = "#%02x%02x%02x" % tuple(int(round(255 * c)) for c in rgb)
        if hexc not in seen:
            seen.add(hexc)
            cols.append(hexc)
        i += 1
    return cols


def set_style():
    """Times New Roman everywhere, TrueType (Type 42) fonts embedded in the PDF."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })
    if not any(f.name == "Times New Roman" for f in font_manager.fontManager.ttflist):
        print("[WARN] 'Times New Roman' not found by matplotlib; falling back to a Times-like serif font. "
              "Install it (or delete ~/.matplotlib/fontlist-*.json to refresh the font cache).")
    return plt


def _style_axes(ax, x, mark_layers):
    if mark_layers:
        a, b = mark_layers
        ax.axvspan(a - 0.5, b + 0.5, color="gray", alpha=0.13, zorder=0)
    ax.set_xlabel("Layer", fontsize=LABEL_FS)
    ax.set_ylabel("JS divergence", fontsize=LABEL_FS)
    ax.set_xticks(x if len(x) <= 12 else x[1::2])
    ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.grid(True, alpha=0.3)


def _subcaption(ax, text):
    """(a)/(b) caption centred under the axes, a fixed distance (in points) below the x-label."""
    ax.annotate(text, xy=(0.5, 0.0), xycoords="axes fraction", xytext=(0, -88),
                textcoords="offset points", ha="center", va="top", fontsize=SUBFIG_FS)


def plot_figure(div, layers, legend_langs, pdf_path, mark_layers=None):
    """Left: JS divergence of every language (+ top-10 spoken highlighted).
    Right: mean +- 1 std (population std, ddof=0) over languages at each layer."""
    plt = set_style()

    langs = list(div)
    n_langs = len(langs)
    legend_langs = [l for l in legend_langs if l in div]
    others = sorted((l for l in langs if l not in legend_langs), key=lambda l: -np.nanmean(div[l]))
    x = np.asarray(layers)

    tab10 = plt.get_cmap("tab10")
    hl_colors = {l: tab10(i % 10) for i, l in enumerate(legend_langs)}
    other_colors = dict(zip(others, distinct_colors(len(others))))

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(20, 7.5))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.2, wspace=0.22)

    # ---- (a) all languages + highlighted top-10 ---------------------------------
    for l in others:
        ax_a.plot(x, div[l], color=other_colors[l], lw=0.8, alpha=0.65, zorder=2)
    for l in legend_langs:
        ax_a.plot(x, div[l], color=hl_colors[l], lw=2.6, marker="o", ms=3.5, zorder=5,
                  label=lang_label(l))
    _style_axes(ax_a, x, mark_layers)
    _subcaption(ax_a, f"(a) {n_langs} languages")

    # ---- (b) mean +- std --------------------------------------------------------
    arr = np.array([div[l] for l in langs], dtype=np.float64)  # (n_langs, L)
    mean, std = np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)
    color = "#1f77b4"
    ax_b.fill_between(x, mean - std, mean + std, color=color, alpha=0.2, linewidth=0, zorder=2,
                      label="\u00b11 std")
    ax_b.plot(x, mean, color=color, lw=3.2, marker="o", ms=4.5, zorder=5, label="Mean")
    _style_axes(ax_b, x, mark_layers)
    _subcaption(ax_b, "(b) Mean \u00b1 std")
    ax_b.legend(loc="upper center", fontsize=LEGEND_FS, frameon=True).get_frame().set_alpha(0.95)

    # ---- legend of the highlighted languages: one row above the panels ---------
    handles, labels = ax_a.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.875),
                   ncol=min(5, len(handles)), fontsize=LEGEND_FS, frameon=True,
                   columnspacing=1.6, handlelength=2.0).get_frame().set_alpha(0.95)

    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def write_csv(path, div, layers):
    with open(path, "w", encoding="utf-8") as f:
        f.write("language,name," + ",".join(f"layer_{l}" for l in layers) + "\n")
        for lang in sorted(div):
            f.write(f"{lang},\"{lang_label(lang)}\"," + ",".join(f"{v:.6f}" for v in div[lang]) + "\n")


def print_layer_summary(div, layers):
    arr = np.array([div[l] for l in div], dtype=np.float64)
    mean = np.nanmean(arr, axis=0)
    print("\nMean divergence over languages, per layer:")
    for l, m in zip(layers, mean):
        print(f"  layer {l:>2}: {m:.4f}  " + "#" * int(round(40 * m / max(mean.max(), 1e-9))))
    print(f"Lowest mean divergence at layer {layers[int(np.argmin(mean))]}")


# ----------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Routing divergence figure over FLORES-200")
    p.add_argument("--model_name_or_path", default="Qwen/Qwen1.5-MoE-A2.7B")
    p.add_argument("--flores_path", default="data/flores.json")
    p.add_argument("--output_dir", default="results/routing_divergence")
    p.add_argument("--num_samples", type=int, default=200,
                   help="#parallel FLORES sentences per language (0 = all). Same sentences for every language.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--languages", nargs="+", default=None, help="subset of FLORES codes (default: all)")
    p.add_argument("--legend_langs", nargs="+", default=DEFAULT_LEGEND_LANGS,
                   help="FLORES codes highlighted in the legend (default: 10 most widely spoken non-English languages)")
    p.add_argument("--mark_layers", type=int, nargs=2, default=None, metavar=("START", "END"),
                   help="optionally shade this 1-indexed inclusive layer range, e.g. 8 16")
    p.add_argument("--plot_only", action="store_true", help="only re-draw from the cached json")
    p.add_argument("--overwrite", action="store_true", help="ignore the cache and recompute everything")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "routing_divergence.json")
    pdf_path = os.path.join(args.output_dir, "routing_divergence.pdf")

    cache = {"meta": {}, "layers": [], "divergence": {}}
    if os.path.exists(json_path) and not args.overwrite:
        with open(json_path, "r", encoding="utf-8") as f:
            cache = json.load(f)

    if not args.plot_only:
        records = load_flores(args.flores_path)
        all_langs = sorted({k for r in records for k in r if LANG_KEY_RE.match(k)})
        if ENGLISH not in all_langs:
            raise KeyError(f"'{ENGLISH}' not found in {args.flores_path}")
        langs = [l for l in (args.languages or all_langs) if l != ENGLISH]
        unknown = [l for l in langs if l not in all_langs]
        if unknown:
            raise KeyError(f"Not in FLORES json: {unknown}")

        pool = [i for i, r in enumerate(records) if r.get(ENGLISH)]
        sample_idx = sorted(random.Random(args.seed).sample(pool, args.num_samples)
                            if 0 < args.num_samples < len(pool) else pool)
        meta = {"model": args.model_name_or_path, "num_samples": len(sample_idx), "seed": args.seed,
                "max_length": args.max_length, "flores": os.path.basename(args.flores_path)}
        if cache["divergence"] and cache["meta"] != meta:
            raise SystemExit(f"{json_path} was made with different settings:\n  cached: {cache['meta']}\n  now:    {meta}\n"
                             "Use another --output_dir or pass --overwrite.")
        cache["meta"] = meta

        todo = [l for l in langs if l not in cache["divergence"]]
        print(f"FLORES: {len(all_langs)} language codes found | {len(sample_idx)} sentences per language | "
              f"{len(langs) - len(todo)} already cached, {len(todo)} to compute")

        if todo:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
            print(f"Loading model: {args.model_name_or_path} (device={device}, dtype={args.dtype})")
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path, trust_remote_code=True, torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None)
            model.eval()
            if device == "cpu":
                model.to(device)

            rec = RoutingRecorder(model)
            cache["layers"] = [li + 1 for li in rec.layer_ids]  # 1-indexed
            print(f"{len(rec.layer_ids)} MoE layers, E={rec.num_experts} experts per layer")

            pos = {i: k for k, i in enumerate(sample_idx)}
            q_en = rec.profiles(tokenizer, [records[i][ENGLISH] for i in sample_idx],
                                device, args.batch_size, args.max_length).astype(np.float64)

            for lang in tqdm(todo, desc="languages"):
                idxs = [i for i in sample_idx if records[i].get(lang)]
                if not idxs:
                    print(f"[WARN] {lang}: no sentences, skipped")
                    continue
                q_l = rec.profiles(tokenizer, [records[i][lang] for i in idxs],
                                   device, args.batch_size, args.max_length).astype(np.float64)
                d = entropy_normalized_jsd(q_en[[pos[i] for i in idxs]], q_l)  # (n, L)
                cache["divergence"][lang] = np.nanmean(d, axis=0).tolist()
                atomic_write_json(json_path, cache)  # checkpoint after every language

    div = {l: np.asarray(v, dtype=np.float64) for l, v in cache["divergence"].items()}
    if not div:
        sys.exit("No divergence results to plot.")
    if args.languages:  # plotting a subset of the cache
        div = {l: v for l, v in div.items() if l in args.languages}
    layers = cache["layers"]

    missing = [l for l in args.legend_langs if l not in div]
    if missing:
        print(f"[WARN] legend languages not available and skipped: {missing}")

    plot_figure(div, layers, args.legend_langs, pdf_path, args.mark_layers)
    write_csv(os.path.join(args.output_dir, "routing_divergence.csv"), div, layers)
    print_layer_summary(div, layers)
    print(f"\nSaved: {pdf_path}\nSaved: {json_path}")


if __name__ == "__main__":
    main()