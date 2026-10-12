"""Chuẩn hóa và làm sạch văn bản (NFKC, ký tự điều khiển, khoảng trắng, lọc trùng chính xác).

Dùng như thư viện (build_*.py import) hoặc CLI:
    python scripts/clean.py IN.jsonl OUT.jsonl [--field text] [--min-chars 50] [--dedup]
"""
from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata
from typing import Optional

from common import log, read_jsonl, setup_logging, write_jsonl

_VT_FF = re.compile(r"[\x0b\x0c]")
_CTRL = re.compile(r"[\x00-\x08\x0e-\x1f\x7f-\x9f]")  # giữ \t \n (\r đã đổi trước)
_SP = re.compile(r"[ \t\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]+")
_NL3 = re.compile(r"\n{3,}")
_WS = re.compile(r"\s+")


def normalize_sentence(s: str) -> str:
    """NFKC + gộp mọi khoảng trắng thành một dấu cách. Dùng cho câu/dữ liệu đánh giá."""
    s = unicodedata.normalize("NFKC", s)
    s = _CTRL.sub("", s.replace("\ufeff", ""))
    return _WS.sub(" ", s).strip()


def normalize_text(s: str) -> str:
    """NFKC cho tài liệu dài: giữ ngắt đoạn, bỏ ký tự điều khiển, gộp khoảng trắng.

    Lưu ý: không xóa ZWJ/ZWNJ vì chúng có nghĩa trong chữ Devanagari/Ả Rập.
    """
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n").replace("\u2028", "\n").replace("\u2029", "\n")
    s = _VT_FF.sub("\n", s.replace("\ufeff", ""))
    s = _CTRL.sub("", s)
    s = "\n".join(_SP.sub(" ", ln).strip() for ln in s.split("\n"))
    return _NL3.sub("\n\n", s).strip()


def clean_document(text: str, min_chars: int = 50, max_replacement_frac: float = 0.01) -> Optional[str]:
    """Trả về văn bản đã chuẩn hóa, hoặc None nếu bị loại (quá ngắn / nhiều ký tự lỗi)."""
    if not text:
        return None
    t = normalize_text(text)
    if len(t) < min_chars:
        return None
    if t.count("\ufffd") > max_replacement_frac * len(t):
        return None
    return t


def text_hash(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


class ExactDeduper:
    """Lọc trùng chính xác theo băm 64-bit của văn bản đã chuẩn hóa."""

    def __init__(self) -> None:
        self.seen: set[int] = set()

    def add(self, text: str) -> bool:
        """True nếu là văn bản mới (đã ghi nhận); False nếu trùng."""
        h = text_hash(text)
        if h in self.seen:
            return False
        self.seen.add(h)
        return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--field", default="text")
    ap.add_argument("--min-chars", type=int, default=50)
    ap.add_argument("--dedup", action="store_true")
    args = ap.parse_args()
    setup_logging()
    ded = ExactDeduper() if args.dedup else None
    stats = {"in": 0, "short": 0, "dup": 0}

    def gen():
        for r in read_jsonl(args.inp):
            stats["in"] += 1
            t = clean_document(r.get(args.field, ""), args.min_chars)
            if t is None:
                stats["short"] += 1
                continue
            if ded and not ded.add(t):
                stats["dup"] += 1
                continue
            r[args.field] = t
            yield r

    n = write_jsonl(args.out, gen())
    log.info("vào=%d ra=%d bỏ_ngắn=%d bỏ_trùng=%d", stats["in"], n, stats["short"], stats["dup"])


if __name__ == "__main__":
    main()
