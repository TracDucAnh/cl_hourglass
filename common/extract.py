"""Trích biểu diễn theo tầng từ một mô hình trên dữ liệu song song (mục 2 và Study 1 của kế hoạch).

Quy ước (đặc tả):
  * Z[i, l, k, :] là biểu diễn của nội dung i viết bằng ngôn ngữ l ở tầng k = 0..K, trong đó k = 0 là đầu ra
    embedding và k = 1..K là đầu ra từng khối, TRƯỚC norm cuối (kể cả tầng cuối). Hook do common.models.LayerCapture
    gắn; ở đây chỉ lo token hóa, gộp, ghi đĩa.
  * Gộp trung bình theo token, bỏ BOS/CLS/SEP/EOS và đệm (mặt nạ = ~token đặc biệt & attention_mask).
  * Lưu fp16, tensor [n_items, L, K+1, d] ghi bằng memmap (Qwen3-8B ~ 3,7 GB); đúng đầu vào của
    common.metrics.suite.compute_curves.

Họ riêng:
  * moe        thêm router_usage.npy [n_items, L, K, E] (tần suất chọn expert, khối dense toàn 0), đầu vào của
               common.metrics.routing.js_routing.
  * diffusion  mỗi câu được che ngẫu nhiên ở các mức t trong mask_levels, n_masks lần che mỗi mức. Mặt nạ lấy từ
               numpy Generator có seed (mask_seed, id nội dung, mức, lần che) nên (i) không phụ thuộc cách chia batch,
               (ii) mọi ngôn ngữ của cùng một nội dung dùng cùng dòng ngẫu nhiên. Mỗi mức t > 0 cho hai biến thể
               gộp: "t{t}_all" (mọi token) và "t{t}_masked" (chỉ token bị che); t = 0 chỉ có "t0.00_all".
               Biểu diễn lưu là trung bình trên n_masks lần che.

Tệp đầu ra trong <out_dir>/<slug(model)>[-randinit]/<dataset>/:
  Z_<variant>.npy   ntok.npy [n_items, L]   router_usage.npy (MoE)   meta.json   progress.json

Chạy lại sẽ tiếp tục từ ngôn ngữ đang dở nếu cấu hình không đổi (progress.json); meta.json có complete=true khi xong.

CLI:
  python -m common.extract --model xlmr-base --dataset flores_devtest --out results/s1/raw
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .io import config_hash, git_commit, load_parallel, set_seed, utcnow
from .models import LayerCapture, LoadedModel, RouterCapture, get_spec, load_model, slug

EXTRACT_VERSION = "1.0"
FP16_MAX = float(np.finfo(np.float16).max)
DEFAULT_MASK_LEVELS: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75)


# ------------------------------------------------------------------ cấu hình
@dataclass
class ExtractConfig:
    batch_size: Optional[int] = None          # None: lấy ModelSpec.batch_size
    max_length: Optional[int] = None          # None: lấy ModelSpec.max_length
    device: Optional[str] = None
    dtype: Optional[str] = None               # None: ModelSpec.dtype
    out_dtype: str = "float16"                # 'float32' chỉ để kiểm thử
    mask_levels: Tuple[float, ...] = DEFAULT_MASK_LEVELS
    n_masks: int = 8
    mask_seed: int = 0
    save_router: bool = True
    from_config: bool = False                 # đối chứng "mô hình khởi tạo ngẫu nhiên"
    init_seed: int = 0
    limit: Optional[int] = None               # chỉ lấy `limit` nội dung đầu (chạy thử)
    overwrite: bool = False


@dataclass
class ExtractResult:
    Z: Dict[str, np.ndarray]                  # biến thể -> [N, K+1, d]
    ntok: np.ndarray                          # [N] số token hợp lệ (không đặc biệt, không đệm)
    usage: Optional[np.ndarray]               # [N, K, E] float32 hoặc None
    n_clipped: int = 0                        # số phần tử vượt phạm vi fp16 bị cắt
    n_nonfinite: int = 0


def variant_names(family: str, mask_levels: Sequence[float] = DEFAULT_MASK_LEVELS) -> List[str]:
    """Danh sách biến thể gộp của một họ mô hình."""
    if family != "diffusion":
        return ["all"]
    names: List[str] = []
    for t in mask_levels:
        if t <= 0:
            names.append("t0.00_all")
        else:
            names += [f"t{t:.2f}_all", f"t{t:.2f}_masked"]
    return names


# ------------------------------------------------------------------ token hóa
def prepare_tokenizer(tok):
    """Đệm bên phải (an toàn cho mô hình nhân quả), và có pad_token."""
    tok.padding_side = "right"
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        elif tok.unk_token is not None:
            tok.pad_token = tok.unk_token
        else:
            raise ValueError("tokenizer không có pad/eos/unk token để làm đệm")
    return tok


def encode_batch(tok, texts: Sequence[str], max_length: int):
    """Trả về (input_ids [B,T], attention_mask [B,T]) dạng torch.long."""
    enc = tok(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    ids = enc["input_ids"]
    att = enc["attention_mask"] if "attention_mask" in enc else (ids != tok.pad_token_id).long()
    return ids, att


def special_token_mask(tok, input_ids):
    """bool [B,T]: True ở token đặc biệt (BOS/CLS/SEP/EOS/pad/...) theo tok.all_special_ids."""
    import torch

    sp = torch.tensor(sorted({int(i) for i in tok.all_special_ids}), dtype=input_ids.dtype)
    return torch.isin(input_ids, sp)


def resolve_mask_id(lm: LoadedModel) -> int:
    """id token [MASK] của mô hình diffusion; cảnh báo nếu ModelSpec và tokenizer không khớp ([XM])."""
    spec_id = lm.spec.mask_token_id
    tok_id = getattr(lm.tokenizer, "mask_token_id", None)
    if spec_id is None and tok_id is None:
        raise ValueError(f"{lm.spec.id}: không biết id token [MASK] (đặt ModelSpec.mask_token_id)")
    if spec_id is not None and tok_id is not None and int(spec_id) != int(tok_id):
        warnings.warn(f"{lm.spec.id}: mask_token_id của spec ({spec_id}) khác tokenizer ({tok_id}); dùng spec")
    return int(spec_id if spec_id is not None else tok_id)


def diffusion_mask(valid: np.ndarray, item_ids: Sequence[int], level_idx: int, mask_idx: int, t: float,
                   mask_seed: int) -> np.ndarray:
    """Mặt nạ che ngẫu nhiên [B,T] (bool) chỉ trên token hợp lệ; số token bị che = max(1, round(t * n_hợp_lệ)).

    Seed theo (mask_seed, id nội dung, level_idx, mask_idx): tất định và không phụ thuộc batch.
    """
    B, T = valid.shape
    out = np.zeros((B, T), dtype=bool)
    for b in range(B):
        pos = np.flatnonzero(valid[b])
        n = len(pos)
        if n == 0:
            continue
        rng = np.random.default_rng([int(mask_seed), int(item_ids[b]), int(level_idx), int(mask_idx)])
        u = rng.random(n)
        m = min(n, max(1, int(round(t * n))))
        out[b, pos[np.argsort(u, kind="stable")[:m]]] = True
    return out


# ------------------------------------------------------------------ forward có hook
_NO_MASK_MODELS: set = set()      # mô hình mà forward không nhận attention_mask (mã tùy biến [XM])


def _forward(lm: LoadedModel, input_ids, attention_mask):
    sid = lm.spec.id
    kw = {"input_ids": input_ids}
    if sid not in _NO_MASK_MODELS:
        kw["attention_mask"] = attention_mask
    try:
        return lm.model(**kw)
    except TypeError as e:
        if "attention_mask" in kw and "attention_mask" in str(e):
            warnings.warn(f"{sid}: forward không nhận attention_mask; chạy không mặt nạ (đệm sẽ được attend). "
                          f"Các batch đã sắp theo độ dài nên đệm ít, nhưng hãy kiểm tra [XM].")
            _NO_MASK_MODELS.add(sid)
            return lm.model(input_ids=input_ids)
        raise


def forward_pooled(lm: LoadedModel, input_ids, attention_mask, weights, *, with_router: bool = False, valid=None):
    """Một lần forward có hook. weights: [P,B,T]. Trả về (pooled [P,B,K+1,d] float32, usage [B,K,E] | None)."""
    import torch

    dev = lm.device
    ids, att, W = input_ids.to(dev), attention_mask.to(dev), weights.to(dev)
    usage = None
    with torch.no_grad():
        if with_router:
            if valid is None:
                raise ValueError("with_router=True cần `valid` [B,T]")
            with LayerCapture(lm) as cap, RouterCapture(lm) as rc:
                cap.set_weights(W)
                _forward(lm, ids, att)
            usage = rc.usage(ids.shape[0], ids.shape[1], valid.to(dev))
        else:
            with LayerCapture(lm) as cap:
                cap.set_weights(W)
                _forward(lm, ids, att)
        pooled = cap.pooled()
    return pooled, usage


# ------------------------------------------------------------------ trích cho một danh sách câu
def extract_texts(lm: LoadedModel, texts: Sequence[str], cfg: Optional[ExtractConfig] = None,
                  item_ids: Optional[Sequence[int]] = None, progress: bool = False) -> ExtractResult:
    """Biểu diễn của `texts` (một ngôn ngữ). Giữ nguyên thứ tự đầu vào dù bên trong sắp theo độ dài.

    item_ids: id số của từng câu (chỉ dùng để dựng mặt nạ diffusion giống nhau giữa các ngôn ngữ);
    mặc định 0..N-1, nên truyền cùng một dãy cho mọi ngôn ngữ.
    """
    import torch

    cfg = cfg or ExtractConfig()
    spec = lm.spec
    tok = prepare_tokenizer(lm.tokenizer)
    N = len(texts)
    ids_num = list(range(N)) if item_ids is None else [int(i) for i in item_ids]
    if len(ids_num) != N:
        raise ValueError("item_ids phải cùng độ dài với texts")
    bs = int(cfg.batch_size or spec.batch_size)
    max_len = int(cfg.max_length or spec.max_length)
    diffusion = spec.family == "diffusion"
    with_router = bool(lm.is_moe and not diffusion and cfg.save_router)
    variants = variant_names(spec.family, cfg.mask_levels)
    out_dt = np.dtype(cfg.out_dtype)
    mask_id = resolve_mask_id(lm) if diffusion else None

    Z = {v: np.zeros((N, lm.K + 1, lm.d), dtype=out_dt) for v in variants}
    ntok = np.zeros(N, dtype=np.int32)
    usage_all = np.zeros((N, lm.K, lm.n_experts), dtype=np.float32) if with_router else None
    stats = {"clip": 0, "nonfinite": 0}

    def store(v: str, idx: np.ndarray, t) -> None:
        arr = t.detach().float().cpu().numpy()
        stats["nonfinite"] += int((~np.isfinite(arr)).sum())
        if out_dt == np.float16:
            stats["clip"] += int((np.abs(arr) > FP16_MAX).sum())
            arr = np.clip(arr, -FP16_MAX, FP16_MAX)
        Z[v][idx] = arr.astype(out_dt)

    lengths = np.array([len(t) for t in texts])
    order = np.argsort(-lengths, kind="stable")                 # câu dài trước: ít đệm, lỗi bộ nhớ lộ sớm
    batches = [order[i:i + bs] for i in range(0, N, bs)]
    it = batches
    if progress:
        try:
            from tqdm import tqdm

            it = tqdm(batches, desc=f"{spec.id}", leave=False)
        except ImportError:
            pass

    for idx in it:
        ids, att = encode_batch(tok, [texts[i] for i in idx], max_len)
        valid = att.bool() & ~special_token_mask(tok, ids)
        ntok[idx] = valid.sum(1).numpy()
        if not diffusion:
            pooled, usage = forward_pooled(lm, ids, att, valid.float()[None], with_router=with_router, valid=valid)
            store("all", idx, pooled[0])
            if usage is not None:
                usage_all[idx] = usage.cpu().numpy()
            continue
        valid_np = valid.numpy()
        batch_ids = [ids_num[i] for i in idx]
        for li, t in enumerate(cfg.mask_levels):
            if t <= 0:
                pooled, _ = forward_pooled(lm, ids, att, valid.float()[None])
                store("t0.00_all", idx, pooled[0])
                continue
            acc = None
            for m in range(cfg.n_masks):
                mk = torch.from_numpy(diffusion_mask(valid_np, batch_ids, li, m, float(t), cfg.mask_seed))
                ids_m = ids.clone()
                ids_m[mk] = mask_id
                W = torch.stack([valid.float(), mk.float()])    # [2,B,T]: mọi token / token bị che
                pooled, _ = forward_pooled(lm, ids_m, att, W)
                acc = pooled if acc is None else acc + pooled
            acc = acc / float(cfg.n_masks)
            store(f"t{t:.2f}_all", idx, acc[0])
            store(f"t{t:.2f}_masked", idx, acc[1])

    if stats["nonfinite"]:
        warnings.warn(f"{spec.id}: {stats['nonfinite']} giá trị không hữu hạn trong biểu diễn")
    return ExtractResult(Z, ntok, usage_all, stats["clip"], stats["nonfinite"])


# ------------------------------------------------------------------ chạy cả bộ dữ liệu, ghi đĩa
def run_dir_for(out_dir, model_id, dataset_stem: str, from_config: bool = False) -> Path:
    spec = get_spec(model_id)
    return Path(out_dir) / (slug(spec.id) + ("-randinit" if from_config else "")) / dataset_stem


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def run_extract(model_id, dataset, langs: Optional[Sequence[str]] = None, out_dir="results/s1/raw",
                cfg: Optional[ExtractConfig] = None, lm: Optional[LoadedModel] = None,
                progress: bool = True) -> Path:
    """Trích biểu diễn của `model_id` trên `dataset` (vd 'flores_devtest') cho `langs` và ghi vào out_dir.

    Trả về thư mục chạy. lm: truyền mô hình đã nạp để dùng lại (bỏ qua load_model).
    """
    cfg = cfg or ExtractConfig()
    spec = get_spec(model_id)
    data = load_parallel(dataset, langs, limit=cfg.limit)
    run = run_dir_for(out_dir, spec, data.dataset, cfg.from_config)
    meta_path, prog_path = run / "meta.json", run / "progress.json"
    if meta_path.exists() and not cfg.overwrite:
        if json.loads(meta_path.read_text(encoding="utf-8")).get("complete"):
            return run
    run.mkdir(parents=True, exist_ok=True)

    if lm is None:
        set_seed(cfg.init_seed)
        lm = load_model(spec, device=cfg.device, dtype=cfg.dtype, from_config=cfg.from_config, seed=cfg.init_seed)
    variants = variant_names(spec.family, cfg.mask_levels)
    N, L, K1, d = len(data), len(data.langs), lm.K + 1, lm.d
    with_router = bool(lm.is_moe and spec.family != "diffusion" and cfg.save_router)

    ident = {"version": EXTRACT_VERSION, "model": spec.id, "hf": spec.hf_name, "dataset": data.dataset,
             "langs": data.langs, "ids": config_hash(data.ids), "cfg": asdict(cfg), "K": lm.K, "d": d}
    chash = config_hash(ident)

    done: List[str] = []
    n_clip = n_nonfin = 0
    resume = False
    if prog_path.exists() and not cfg.overwrite:
        pr = json.loads(prog_path.read_text(encoding="utf-8"))
        files_ok = all((run / f"Z_{v}.npy").exists() for v in variants) and (run / "ntok.npy").exists()
        if pr.get("config_hash") == chash and files_ok:
            done, n_clip, n_nonfin = list(pr["done_langs"]), int(pr["n_clipped"]), int(pr["n_nonfinite"])
            resume = True
    mode = "r+" if resume else "w+"
    dt = np.dtype(cfg.out_dtype)
    mm = {v: np.lib.format.open_memmap(run / f"Z_{v}.npy", mode=mode, dtype=dt, shape=(N, L, K1, d))
          for v in variants}
    ntok = np.load(run / "ntok.npy") if resume else np.zeros((N, L), dtype=np.int32)
    usage_mm = None
    if with_router:
        usage_mm = np.lib.format.open_memmap(run / "router_usage.npy", mode=mode, dtype=np.float32,
                                             shape=(N, L, lm.K, lm.n_experts))

    t_start = time.time()
    for li, lang in enumerate(data.langs):
        if lang in done:
            continue
        res = extract_texts(lm, data.texts[lang], cfg, item_ids=range(N), progress=progress)
        for v in variants:
            mm[v][:, li] = res.Z[v]
            mm[v].flush()
        if usage_mm is not None:
            usage_mm[:, li] = res.usage
            usage_mm.flush()
        ntok[:, li] = res.ntok
        np.save(run / "ntok.npy", ntok)
        n_clip += res.n_clipped
        n_nonfin += res.n_nonfinite
        done.append(lang)
        _write_json(prog_path, {"config_hash": chash, "done_langs": done, "n_clipped": n_clip,
                                "n_nonfinite": n_nonfin, "updated_at": utcnow()})

    meta = {
        "version": EXTRACT_VERSION, "complete": True, "model_id": spec.id, "hf_name": spec.hf_name,
        "family": spec.family, "arch": spec.arch, "random_init": bool(cfg.from_config), "K": lm.K, "d": d,
        "dataset": data.dataset, "langs": data.langs, "ids": data.ids, "n_items": N, "variants": variants,
        "default_variant": "all" if "all" in variants else variants[0], "shape": [N, L, K1, d],
        "dtype": cfg.out_dtype, "n_clipped": n_clip, "n_nonfinite": n_nonfin,
        "n_experts": lm.n_experts if with_router else None, "top_k": lm.top_k if with_router else None,
        "mask_levels": list(cfg.mask_levels) if spec.family == "diffusion" else None,
        "n_masks": cfg.n_masks if spec.family == "diffusion" else None,
        "mask_seed": cfg.mask_seed if spec.family == "diffusion" else None,
        "ntok_mean": {l: float(ntok[:, i].mean()) for i, l in enumerate(data.langs)},
        "config_hash": chash, "git_commit": git_commit(), "created_at": utcnow(),
        "seconds_this_run": round(time.time() - t_start, 1),
    }
    try:
        import torch
        import transformers

        meta["torch"], meta["transformers"] = torch.__version__, transformers.__version__
    except ImportError:
        pass
    _write_json(meta_path, meta)
    return run


# ------------------------------------------------------------------ đọc lại
def load_meta(run_dir) -> dict:
    return json.loads((Path(run_dir) / "meta.json").read_text(encoding="utf-8"))


def load_extract(run_dir, variant: Optional[str] = None, mmap: bool = True) -> Tuple[np.ndarray, dict]:
    """(Z [n_items, L, K+1, d], meta). variant=None lấy meta['default_variant']."""
    meta = load_meta(run_dir)
    variant = variant or meta["default_variant"]
    if variant not in meta["variants"]:
        raise KeyError(f"biến thể {variant!r} không có; có {meta['variants']}")
    Z = np.load(Path(run_dir) / f"Z_{variant}.npy", mmap_mode="r" if mmap else None)
    return Z, meta


def load_router_usage(run_dir, mmap: bool = True) -> np.ndarray:
    """router_usage [n_items, L, K, E]; dùng js_routing(U[:, la], U[:, lb])."""
    p = Path(run_dir) / "router_usage.npy"
    if not p.exists():
        raise FileNotFoundError(f"{p}: không phải MoE hoặc chưa lưu router")
    return np.load(p, mmap_mode="r" if mmap else None)


def load_ntok(run_dir) -> np.ndarray:
    """Số token hợp lệ [n_items, L] (để tính fertility / token mỗi từ)."""
    return np.load(Path(run_dir) / "ntok.npy")


# ------------------------------------------------------------------ CLI
def _main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Trích biểu diễn theo tầng (hook trước norm cuối, gộp token)")
    ap.add_argument("--model", required=True, help="id trong registry (xem python -m common.models --list)")
    ap.add_argument("--dataset", default="flores_devtest")
    ap.add_argument("--langs", default=None, help="danh sách mã ngôn ngữ phân tách bằng dấu phẩy; mặc định: mọi ngôn ngữ trong tệp")
    ap.add_argument("--out", default="results/s1/raw")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--from-config", action="store_true", help="đối chứng khởi tạo ngẫu nhiên")
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--mask-levels", default=",".join(str(t) for t in DEFAULT_MASK_LEVELS))
    ap.add_argument("--n-masks", type=int, default=8)
    ap.add_argument("--mask-seed", type=int, default=0)
    ap.add_argument("--no-router", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    a = ap.parse_args(argv)
    cfg = ExtractConfig(
        batch_size=a.batch_size, max_length=a.max_length, device=a.device, dtype=a.dtype,
        mask_levels=tuple(float(x) for x in a.mask_levels.split(",") if x != ""), n_masks=a.n_masks,
        mask_seed=a.mask_seed, save_router=not a.no_router, from_config=a.from_config,
        init_seed=a.init_seed, limit=a.limit, overwrite=a.overwrite,
    )
    langs = [x for x in a.langs.split(",") if x] if a.langs else None
    print(run_extract(a.model, a.dataset, langs, a.out, cfg, progress=not a.no_progress))


if __name__ == "__main__":
    _main()