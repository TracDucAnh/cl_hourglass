"""Registry mô hình: id -> cách nạp, số tầng K, chiều d, và cách gắn hook (dense / moe / diffusion / encoder / charbyte).

Hai lớp tách bạch:
  * ModelSpec    mô tả MỘT mô hình (tên HF, họ, kiến trúc, token mask cho diffusion, ...). Danh sách mô hình của
                 Study 1 nằm ở REGISTRY bên dưới; thêm mô hình mới = thêm một ModelSpec (hoặc gọi register()).
  * ArchAdapter  mô tả MỘT kiến trúc: module nào cho "biểu diễn tầng 0" (đầu ra embedding), ModuleList nào chứa K
                 khối, module norm cuối (chỉ để kiểm thử), module router (MoE). Dùng chung cho nhiều mô hình.

Quy ước biểu diễn (đặc tả): hook đầu ra của embedding (k = 0) và đầu ra của từng khối (k = 1..K), TRƯỚC norm cuối,
kể cả tầng cuối. output_hidden_states của HF áp norm cuối lên tầng cuối nên KHÔNG dùng được cho mục đích này.

Các tên trong REGISTRY và đường dẫn module của mô hình dùng mã tùy biến (LLaDA, Dream, DeepSeek-V2) được viết theo
trí nhớ về cấu trúc mã nguồn của chúng và đánh dấu [XM]; hãy chạy `python -m common.models --check <id>` trên máy
có mạng trước khi chạy lưới (xem verify_attached).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ------------------------------------------------------------------ mô tả kiến trúc
PREFIXES = ("", "model.", "language_model.", "model.language_model.", "transformer.")


@dataclass(frozen=True)
class ArchAdapter:
    name: str
    embed: Tuple[str, ...]                 # ứng viên đường dẫn tới module có đầu ra = biểu diễn tầng 0
    blocks: Tuple[str, ...]                # ứng viên đường dẫn tới ModuleList K khối
    final_norm: Tuple[str, ...] = ()       # norm cuối (chỉ dùng để kiểm thử hook tầng cuối)
    gate: Tuple[str, ...] = ()             # tên module router bên trong mỗi khối (MoE)
    mask_adapter: Optional[str] = None     # 'canine': đầu ra các khối sâu có độ dài chuỗi khác embedding


ARCHS: Dict[str, ArchAdapter] = {
    # BERT, XLM-R (post-LN: không có norm cuối riêng, LayerNorm nằm trong từng khối)
    "bert": ArchAdapter("bert", embed=("embeddings",), blocks=("encoder.layer",)),
    # Llama, Qwen3, Gemma3 (văn bản), Dream (kiến trúc Qwen2.5), DeepSeek-V2: embed_tokens / layers / norm
    "llama": ArchAdapter("llama", embed=("embed_tokens",), blocks=("layers",), final_norm=("norm",),
                         gate=("mlp.gate", "block_sparse_moe.gate", "mlp.router")),
    "bloom": ArchAdapter("bloom", embed=("word_embeddings_layernorm",), blocks=("h",), final_norm=("ln_f",)),
    # LLaDA (kiến trúc OLMo): model.transformer.{wte, blocks, ln_f}   [XM]
    "llada": ArchAdapter("llada", embed=("model.transformer.wte", "transformer.wte"),
                         blocks=("model.transformer.blocks", "transformer.blocks"),
                         final_norm=("model.transformer.ln_f", "transformer.ln_f")),
    # T5EncoderModel (ByT5): encoder.{embed_tokens, block, final_layer_norm}
    "t5enc": ArchAdapter("t5enc", embed=("encoder.embed_tokens",), blocks=("encoder.block",),
                         final_norm=("encoder.final_layer_norm",)),
    # CANINE: biểu diễn tầng 0 là embedding ký tự; K khối sâu chạy trên chuỗi đã giảm mẫu 4 lần
    "canine": ArchAdapter("canine", embed=("char_embeddings",), blocks=("encoder.layer",), mask_adapter="canine"),
}


# ------------------------------------------------------------------ mô tả mô hình
@dataclass(frozen=True)
class ModelSpec:
    id: str
    hf_name: str
    family: str                       # 'encoder' | 'dense' | 'moe' | 'diffusion' | 'charbyte'
    arch: str                         # khóa của ARCHS
    auto_class: str = "AutoModel"     # 'AutoModel' | 'T5EncoderModel' | 'CanineModel'
    trust_remote_code: bool = False
    mask_token_id: Optional[int] = None   # diffusion: id token [MASK]; None = lấy từ tokenizer.mask_token_id
    max_length: int = 512
    batch_size: int = 32
    dtype: str = "bfloat16"
    notes: str = ""

    def with_(self, **kw) -> "ModelSpec":
        return replace(self, **kw)


def _S(id_, hf, family, arch, **kw) -> ModelSpec:
    return ModelSpec(id_, hf, family, arch, **kw)


# Danh sách của Study 1 (mục 4 của kế hoạch). [XM] mọi tên HF cần xác minh trước khi chạy.
_SPECS: List[ModelSpec] = [
    # --- encoder
    _S("mbert", "bert-base-multilingual-cased", "encoder", "bert"),
    _S("xlmr-base", "xlm-roberta-base", "encoder", "bert"),
    _S("xlmr-large", "xlm-roberta-large", "encoder", "bert", batch_size=16),
    # --- ký tự / byte
    _S("canine-s", "google/canine-s", "charbyte", "canine", auto_class="CanineModel", max_length=2048, batch_size=16),
    _S("byt5-base-enc", "google/byt5-base", "charbyte", "t5enc", auto_class="T5EncoderModel", max_length=1024,
       batch_size=16, notes="chỉ encoder của ByT5"),
    # --- dense decoder-only
    _S("qwen3-0.6b", "Qwen/Qwen3-0.6B-Base", "dense", "llama"),
    _S("qwen3-1.7b", "Qwen/Qwen3-1.7B-Base", "dense", "llama"),
    _S("qwen3-4b", "Qwen/Qwen3-4B-Base", "dense", "llama", batch_size=16),
    _S("qwen3-8b", "Qwen/Qwen3-8B-Base", "dense", "llama", batch_size=8),
    _S("bloom-560m", "bigscience/bloom-560m", "dense", "bloom"),
    _S("bloom-1b7", "bigscience/bloom-1b7", "dense", "bloom"),
    # --- MoE
    _S("olmoe-1b-7b", "allenai/OLMoE-1B-7B-0924", "moe", "llama", batch_size=16),
    _S("qwen1.5-moe-a2.7b", "Qwen/Qwen1.5-MoE-A2.7B", "moe", "llama", batch_size=8),
    _S("deepseek-v2-lite", "deepseek-ai/DeepSeek-V2-Lite", "moe", "llama", trust_remote_code=True, batch_size=8,
       notes="[XM] mã tùy biến; tầng 0 là dense (không có router); gate trả (topk_idx, topk_weight, aux)"),
    # --- diffusion
    _S("llada-8b", "GSAI-ML/LLaDA-8B-Base", "diffusion", "llada", trust_remote_code=True, mask_token_id=126336,
       batch_size=4, notes="[XM] mask id 126336 theo trí nhớ; kiểm tokenizer"),
    _S("dream-7b", "Dream-org/Dream-v0-Base-7B", "diffusion", "llama", trust_remote_code=True, mask_token_id=151666,
       batch_size=4, notes="[XM] mask id 151666 theo trí nhớ; kiểm tokenizer"),
    # --- giữ ra ngoài cho F1 (Study 5)
    _S("gemma3-1b", "google/gemma-3-1b-pt", "dense", "llama"),
    _S("gemma3-4b", "google/gemma-3-4b-pt", "dense", "llama", batch_size=16,
       notes="[XM] checkpoint đa phương thức; đường dẫn language_model.* do PREFIXES xử lý"),
    _S("llama3.2-1b", "meta-llama/Llama-3.2-1B", "dense", "llama"),
    _S("llama3.2-3b", "meta-llama/Llama-3.2-3B", "dense", "llama", batch_size=16),
]

REGISTRY: Dict[str, ModelSpec] = {s.id: s for s in _SPECS}


def register(spec: ModelSpec, overwrite: bool = False) -> ModelSpec:
    if spec.arch not in ARCHS:
        raise KeyError(f"arch {spec.arch!r} không có trong ARCHS ({sorted(ARCHS)})")
    if spec.id in REGISTRY and not overwrite:
        raise KeyError(f"{spec.id!r} đã có; dùng overwrite=True để ghi đè")
    REGISTRY[spec.id] = spec
    return spec


def get_spec(model_id: str | ModelSpec) -> ModelSpec:
    """Tra theo id ngắn hoặc tên HF."""
    if isinstance(model_id, ModelSpec):
        return model_id
    if model_id in REGISTRY:
        return REGISTRY[model_id]
    for s in REGISTRY.values():
        if s.hf_name == model_id:
            return s
    raise KeyError(f"không có mô hình {model_id!r}; có: {sorted(REGISTRY)}")


def list_models(family: Optional[str] = None) -> List[str]:
    return [k for k, s in REGISTRY.items() if family is None or s.family == family]


def slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)


# ------------------------------------------------------------------ tìm module theo đường dẫn
def _getattr_path(obj, path: str):
    for part in path.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def resolve_module(model, candidates: Sequence[str]):
    """Thử mọi tổ hợp (tiền tố, ứng viên); trả về (đường dẫn đầy đủ, module) đầu tiên tìm thấy."""
    for pre in PREFIXES:
        for c in candidates:
            m = _getattr_path(model, pre + c) if pre else _getattr_path(model, c)
            if m is not None:
                return pre + c, m
    return None, None


def moe_dims(config) -> Tuple[Optional[int], Optional[int]]:
    """(số expert, top_k) đọc từ config HF; None nếu không phải MoE."""
    cfgs = [config] + [getattr(config, a, None) for a in ("text_config", "language_config")]
    for c in cfgs:
        if c is None:
            continue
        e = next((getattr(c, a) for a in ("num_experts", "num_local_experts", "n_routed_experts")
                  if getattr(c, a, None)), None)
        k = getattr(c, "num_experts_per_tok", None)
        if e and k:
            return int(e), int(k)
    return None, None


def hidden_size(model) -> Optional[int]:
    cfgs = [model.config] + [getattr(model.config, a, None) for a in ("text_config",)]
    for c in cfgs:
        if c is None:
            continue
        for a in ("hidden_size", "d_model", "n_embd"):
            v = getattr(c, a, None)
            if v:
                return int(v)
    return None


# ------------------------------------------------------------------ mô hình đã gắn
@dataclass
class LoadedModel:
    spec: ModelSpec
    model: object
    tokenizer: object
    embed: object
    blocks: list
    final_norm: Optional[object]
    gates: list                       # mỗi khối một module router hoặc None
    K: int
    d: int
    paths: Dict[str, str] = field(default_factory=dict)
    n_experts: Optional[int] = None
    top_k: Optional[int] = None

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def is_moe(self) -> bool:
        return any(g is not None for g in self.gates)


def attach(model, tokenizer, spec: ModelSpec | str) -> LoadedModel:
    """Gắn một mô hình HF ĐÃ NẠP vào registry: tìm embed, K khối, norm cuối và router theo ArchAdapter."""
    spec = get_spec(spec)
    ad = ARCHS[spec.arch]
    p_e, embed = resolve_module(model, ad.embed)
    p_b, blocks = resolve_module(model, ad.blocks)
    if embed is None or blocks is None:
        raise AttributeError(
            f"{spec.id} (arch={spec.arch}): không tìm thấy "
            f"{'embed ' + str(ad.embed) if embed is None else ''}{' ' if embed is None and blocks is None else ''}"
            f"{'blocks ' + str(ad.blocks) if blocks is None else ''}. "
            f"Sửa ARCHS['{spec.arch}'] hoặc đăng ký arch mới. Các con của model: {[n for n, _ in model.named_children()]}"
        )
    blocks = list(blocks)
    p_n, fnorm = resolve_module(model, ad.final_norm) if ad.final_norm else (None, None)
    gates = []
    for blk in blocks:
        g = None
        for name in ad.gate:
            g = _getattr_path(blk, name)
            if g is not None:
                break
        gates.append(g)
    n_exp, top_k = moe_dims(model.config)
    if spec.family == "moe" and not any(g is not None for g in gates):
        raise AttributeError(f"{spec.id}: family='moe' nhưng không thấy router trong khối nào (thử ARCHS[...].gate)")
    d = hidden_size(model)
    if d is None:
        d = int(embed.weight.shape[-1]) if hasattr(embed, "weight") else -1
    return LoadedModel(spec, model, tokenizer, embed, blocks, fnorm, gates, len(blocks), int(d),
                       {"embed": p_e, "blocks": p_b, "final_norm": p_n}, n_exp, top_k)


def load_model(spec: ModelSpec | str, *, device: Optional[str] = None, dtype: Optional[str] = None,
               from_config: bool = False, seed: int = 0, local_files_only: bool = False) -> LoadedModel:
    """Tải trọng số + tokenizer từ Hugging Face rồi attach().

    from_config=True: chỉ tải config và khởi tạo NGẪU NHIÊN (đối chứng "mô hình khởi tạo ngẫu nhiên" của Study 1).
    """
    import torch
    import transformers as T

    from .io import set_seed

    spec = get_spec(spec)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dt = getattr(torch, dtype or spec.dtype)
    if dev == "cpu" and dt in (torch.float16, torch.bfloat16):
        dt = torch.float32                     # CPU: tránh phép tính nửa độ chính xác chậm/sai
    kw = dict(trust_remote_code=spec.trust_remote_code, local_files_only=local_files_only)
    cls = getattr(T, spec.auto_class)
    tok = T.AutoTokenizer.from_pretrained(spec.hf_name, **kw)
    if from_config:
        set_seed(seed)
        cfg = T.AutoConfig.from_pretrained(spec.hf_name, **kw)
        model = cls.from_config(cfg, **({"trust_remote_code": True} if spec.trust_remote_code else {})) \
            if spec.auto_class == "AutoModel" else cls(cfg)
        model = model.to(dt)
    else:
        model = cls.from_pretrained(spec.hf_name, torch_dtype=dt, **kw)
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return attach(model, tok, spec)


# ------------------------------------------------------------------ hook gộp token theo từng tầng
def _first_tensor(out):
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


class LayerCapture:
    """Context manager gắn forward hook lên embed + K khối và gộp token (trung bình có trọng số) NGAY trong hook.

    Cách dùng:
        with LayerCapture(lm) as cap:
            cap.set_weights(W)          # W: [P, B, T] trọng số gộp (P cách gộp khác nhau, ví dụ "mọi token", "token bị che")
            lm.model(**inputs)
        pooled = cap.pooled()           # [P, B, K+1, d] float32; tầng 0 = embed, tầng k = đầu ra khối k, TRƯỚC norm cuối
    keep_hidden=True: giữ nguyên tensor [B, T', d] của từng tầng (dùng cho kiểm thử, tốn bộ nhớ).
    """

    def __init__(self, lm: LoadedModel, keep_hidden: bool = False):
        self.lm = lm
        self.keep_hidden = keep_hidden
        self._handles = []
        self._W = None
        self._out: List[Optional[object]] = [None] * (lm.K + 1)
        self.hidden: List[Optional[object]] = [None] * (lm.K + 1)
        self._adapter = lm.spec.arch and ARCHS[lm.spec.arch].mask_adapter

    def set_weights(self, W) -> None:
        if W.dim() != 3:
            raise ValueError("W phải có dạng [P, B, T]")
        self._W = W
        self._out = [None] * (self.lm.K + 1)
        self.hidden = [None] * (self.lm.K + 1)

    def _weights_for(self, T_out: int):
        W = self._W
        if W.shape[-1] == T_out:
            return W
        if self._adapter == "canine":
            return canine_downsample_weights(W, T_out)
        raise RuntimeError(f"độ dài chuỗi đầu ra tầng ({T_out}) khác độ dài đầu vào ({W.shape[-1]}) và kiến trúc "
                           f"{self.lm.spec.arch} không có mask_adapter")

    def _make_hook(self, k: int):
        import torch

        def hook(_module, _inp, out):
            h = _first_tensor(out)
            if self._W is None:
                raise RuntimeError("chưa gọi set_weights() trước forward")
            if h.dim() != 3:
                raise RuntimeError(f"tầng {k}: hook nhận tensor {tuple(h.shape)}, cần [B, T, d]")
            w = self._weights_for(h.shape[1]).to(device=h.device, dtype=torch.float32)       # [P, B, T]
            hf = h.detach().float()
            num = torch.einsum("pbt,btd->pbd", w, hf)
            den = w.sum(-1, keepdim=True).clamp_min(1e-12)
            self._out[k] = num / den
            if self.keep_hidden:
                self.hidden[k] = hf
        return hook

    def __enter__(self):
        mods = [self.lm.embed] + list(self.lm.blocks)
        for k, m in enumerate(mods):
            self._handles.append(m.register_forward_hook(self._make_hook(k)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    def pooled(self):
        import torch

        missing = [k for k, o in enumerate(self._out) if o is None]
        if missing:
            raise RuntimeError(f"hook không chạy ở các tầng {missing}: kiểm tra attach()/ARCHS cho {self.lm.spec.id}")
        return torch.stack(self._out, dim=2)                 # [P, B, K+1, d]


def canine_downsample_weights(W, T_out: int):
    """Trọng số gộp cho các khối sâu của CANINE (chuỗi ký tự giảm mẫu 4 lần).

    Cùng quy tắc với CanineModel._downsample_attention_mask: max-pool kernel = stride = 4 trên mặt nạ, rồi
    chuẩn bị vị trí [CLS] đứng riêng ở đầu. [XM] kiểm lại bằng test_canine_downsample trên transformers đang dùng.
    """
    import torch
    import torch.nn.functional as F

    P, B, T = W.shape
    pooled = F.max_pool1d(W.reshape(P * B, 1, T).float(), kernel_size=4, stride=4).squeeze(1)   # [PB, T//4]
    pooled = pooled.reshape(P, B, -1)
    if pooled.shape[-1] != T_out:
        raise RuntimeError(f"CANINE: độ dài sau giảm mẫu {pooled.shape[-1]} khác {T_out}")
    return pooled


# ------------------------------------------------------------------ hook router (MoE)
def parse_router_output(out, n_experts: int, top_k: int):
    """Chuẩn hóa đầu ra của module router về chỉ số top-k [N, k].

    Chấp nhận: logits [N, E] (float); hoặc tuple chứa tensor số nguyên [N, k] (chỉ số expert), như router của
    DeepSeek-V2 (topk_idx, topk_weight, aux) hoặc router "TopK" ở các bản transformers mới.
    """
    import torch

    cands = out if isinstance(out, (tuple, list)) else (out,)
    for t in cands:
        if torch.is_tensor(t) and t.dim() == 2 and not t.dtype.is_floating_point and t.shape[1] == top_k:
            return t.long()
    for t in cands:
        if torch.is_tensor(t) and t.dim() == 2 and t.dtype.is_floating_point and t.shape[1] == n_experts:
            return torch.topk(t.float(), top_k, dim=-1).indices
    raise RuntimeError("không nhận ra đầu ra của router: cần logits [N, E] hoặc chỉ số nguyên [N, top_k]")


class RouterCapture:
    """Hook các router để lấy top-k expert của từng token ở từng khối MoE (khối dense cho None)."""

    def __init__(self, lm: LoadedModel):
        if not lm.is_moe:
            raise ValueError(f"{lm.spec.id} không phải MoE")
        if lm.n_experts is None or lm.top_k is None:
            raise ValueError("không đọc được số expert / top_k từ config")
        self.lm = lm
        self._handles = []
        self.topk: List[Optional[object]] = [None] * lm.K

    def __enter__(self):
        for k, g in enumerate(self.lm.gates):
            if g is None:
                continue

            def hook(_m, _i, out, k=k):
                self.topk[k] = parse_router_output(out, self.lm.n_experts, self.lm.top_k)
            self._handles.append(g.register_forward_hook(hook))
        self.topk = [None] * self.lm.K
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    def usage(self, B: int, T: int, valid):
        """Tần suất chọn expert [B, K, E] (chuẩn hóa tổng 1 trên token hợp lệ); khối dense toàn 0.

        valid: [B, T] bool, token được tính (không đệm, không đặc biệt).
        """
        import torch

        E = self.lm.n_experts
        out = torch.zeros(B, self.lm.K, E, dtype=torch.float32, device=valid.device)
        v = valid.to(torch.float32)
        for k, idx in enumerate(self.topk):
            if idx is None:
                continue
            idx = idx.to(valid.device).reshape(B, T, -1)
            oh = torch.nn.functional.one_hot(idx, E).sum(2).float()          # [B, T, E]
            cnt = (oh * v[..., None]).sum(1)                                  # [B, E]
            out[:, k] = cnt / cnt.sum(-1, keepdim=True).clamp_min(1e-12)
        return out


# ------------------------------------------------------------------ kiểm tra nhanh trên một mô hình đã gắn
def verify_attached(lm: LoadedModel, text: str = "Hello world. Xin chào thế giới.") -> Dict[str, object]:
    """Chạy 1 câu và xác nhận: hook chạy ở đủ K+1 tầng, chiều d khớp, router (nếu MoE) trả top-k đúng.
    Dùng cho `python -m common.models --check <id>` trên máy thật (mô hình tùy biến LLaDA, Dream, DeepSeek-V2 [XM])."""
    import torch

    tok = lm.tokenizer
    enc = tok([text], return_tensors="pt")
    ids = enc["input_ids"].to(lm.device)
    T = ids.shape[1]
    W = torch.ones(1, 1, T, device=lm.device)
    with torch.no_grad(), LayerCapture(lm) as cap:
        cap.set_weights(W)
        lm.model(input_ids=ids, **({"attention_mask": enc["attention_mask"].to(lm.device)}
                                   if "attention_mask" in enc else {}))
    z = cap.pooled()
    rep = {"K": lm.K, "d": lm.d, "pooled_shape": tuple(z.shape), "finite": bool(torch.isfinite(z).all())}
    if z.shape[-1] != lm.d:
        raise AssertionError(f"d khai báo {lm.d} nhưng hook cho {z.shape[-1]}")
    if lm.is_moe:
        with torch.no_grad(), RouterCapture(lm) as rc:
            lm.model(input_ids=ids)
        rep["moe_layers"] = [k for k, t in enumerate(rc.topk) if t is not None]
    return rep


def _main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Kiểm tra registry trên máy có mạng")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", metavar="ID", help="nạp mô hình thật và chạy verify_attached")
    ap.add_argument("--from-config", action="store_true", help="chỉ tải config, khởi tạo ngẫu nhiên")
    a = ap.parse_args(argv)
    if a.list or not a.check:
        for k, s in REGISTRY.items():
            print(f"{k:22s} {s.family:9s} {s.arch:7s} {s.hf_name}")
        return
    lm = load_model(a.check, from_config=a.from_config)
    print(verify_attached(lm))


if __name__ == "__main__":
    _main()