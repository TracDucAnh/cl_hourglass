# data/ — dữ liệu dùng chung cho kế hoạch LE-D (Hourglass đa ngữ)

Thư mục này cài đặt phần `data/` trong mục 3 của kế hoạch: **chỉ chứa code và các thư mục rỗng**.
Dữ liệu thật được tải/xây ngay trên máy bạn bằng các script dưới đây. Quy tắc của kế hoạch được giữ nguyên:
dữ liệu chỉ đọc sau khi build; mọi bộ trong `processed/` sinh từ `scripts/` và có dòng trong `manifest.json`.

## Cấu trúc

```
data/
├── config.yaml            # ngôn ngữ, mã theo từng bộ dữ liệu, ngân sách byte, nguồn   (bổ sung so với đặc tả)
├── requirements.txt
├── manifest.json          # nguồn, phiên bản, checksum (scripts tự ghi)
├── raw/                   # tải nguyên bản
│   ├── flores/ ntrex/ fineweb2/ fineweb/ opus100/ xnli/ ud/ belebele/ muse/ glotlid/
│   └── wals/                                                                          # (bổ sung: WALS cho processed/typology)
├── processed/
│   ├── parallel/          # flores_dev.jsonl, flores_devtest.jsonl, ntrex.jsonl, opus100_en-xx.jsonl   {id, lang, text}
│   ├── mono/              # {lang}/shard_*.jsonl + _meta.json ; byte_premium.json
│   ├── tasks/             # xnli/, upos/, belebele/
│   └── typology/          # lang2vec_*.csv, wals_*.csv, wordfreq/{lang}.tsv
└── scripts/
    ├── common.py          # đường dẫn, config, jsonl, manifest, tải tệp, ngân sách byte
    ├── fetch.py           # tải raw/ + --verify cho các mục [XM]
    ├── clean.py           # NFKC, ký tự điều khiển, khoảng trắng, lọc trùng chính xác
    ├── leakage_check.py   # chỉ mục 13-gram FLORES/NTREX/XNLI/UD, lọc tài liệu rò rỉ
    ├── byte_premium.py    # prem_l = byte_l / byte_eng trên FLORES devtest + kế hoạch ngân sách
    ├── build_mono.py      # raw FineWeb(-2) -> shard theo ngân sách byte
    ├── build_parallel.py  # (bổ sung) FLORES / NTREX / OPUS-100 -> processed/parallel
    ├── build_tasks.py     # (bổ sung) XNLI, UPOS (UD), Belebele -> processed/tasks
    ├── build_typology.py  # (bổ sung) lang2vec, WALS, bảng tần suất từ
    └── selftest.py        # (bổ sung) kiểm thử offline toàn pipeline trên dữ liệu giả
```

Đặc tả liệt kê 5 script (`fetch, clean, leakage_check, byte_premium, build_mono`) nhưng cũng yêu cầu `processed/parallel`,
`tasks`, `typology`; ba script `build_*` còn lại được thêm để lấp phần đó.

## Cài đặt

```bash
cd data
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/selftest.py                              # ~1 phút, offline; phải in "TẤT CẢ ... KIỂM TRA ĐỀU QUA"
```

Lệnh nào cũng chạy từ thư mục `data/` (hoặc bất kỳ đâu: script tự định vị theo vị trí của nó).
Nếu cần, đặt `export HF_TOKEN=hf_...` (Windows: `set HF_TOKEN=...`) để tải Hugging Face ổn định hơn.

## Chạy theo thứ tự

**Bước 0 — xác minh các mục [XM] (không tải dữ liệu):**
```bash
python scripts/fetch.py --verify          # kiểm tra URL / repo / tên config; kết quả ghi vào manifest.json
```
Dòng `[LỖI]` cho MUSE/swh là bình thường (MUSE không có tiếng Swahili). Lỗi khác: sửa `config.yaml` rồi chạy lại.

**Bước 1 — dữ liệu đánh giá và nhẹ (nhóm `small`):**
```bash
python scripts/fetch.py --skip glotlid    # flores ntrex xnli ud belebele muse wals  (glotlid có thể rất nặng, tải riêng khi cần)
python scripts/fetch.py glotlid           # tuỳ chọn, cho Study 3 (độ trung thành ngôn ngữ đầu ra)

python scripts/build_parallel.py          # -> processed/parallel/flores_*.jsonl, ntrex.jsonl
python scripts/build_tasks.py             # -> processed/tasks/{xnli,upos,belebele}
python scripts/leakage_check.py build     # -> processed/leakage_index.npz (13-gram của FLORES, NTREX, XNLI, UD)
python scripts/byte_premium.py            # -> processed/mono/byte_premium.json, in kế hoạch ngân sách + dung lượng
python scripts/build_typology.py          # lang2vec + WALS
```
Đến đây Study 1 (suy luận trên mô hình mở) đã có đủ dữ liệu (`processed/parallel/flores_devtest.jsonl`).

**Bước 2 — dữ liệu huấn luyện đơn ngữ (nhóm `large`):** thử nhỏ trước!
```bash
python scripts/fetch.py fineweb2 --scale 0.01          # 1% ngân sách: vài trăm MB, để kiểm tra đường ống
python scripts/build_mono.py --scale 0.01
#  hài lòng rồi thì làm thật (xoá raw/fineweb*/ và processed/mono/<lang>/ của lần thử, hoặc thêm --force):
python scripts/fetch.py fineweb2
python scripts/build_mono.py                           # có thể chạy song song: --langs fra zho  /  --langs rus tur ...
python scripts/build_typology.py wordfreq              # tần suất từ cho 12 ngôn ngữ (Study 4)
```
`fetch.py fineweb2` bắt buộc đã có `byte_premium.json` (Bước 1); hoặc dùng `--bytes-per-lang N` để bỏ qua.
`--scale` phải giống nhau ở `fetch.py` và `build_mono.py`.

**Bước 3 — nhánh phụ π = 1% (OPUS-100, chỉ L = 4):**
```bash
python scripts/fetch.py opus100
python scripts/build_parallel.py opus100               # nên chạy sau leakage_check.py build để lọc rò rỉ
```

## Dung lượng và thời gian (ước tính)

| Phần | Dung lượng | Ghi chú |
|---|---|---|
| `small` (trừ GlotLID) | vài trăm MB | UD tải zip rồi xoá, chỉ giữ tệp train/dev/test |
| `fineweb2` raw | ≈ 14 GB chưa nén (≈ 5–7 GB đĩa, vì gzip) | = ngân sách × `raw_margin` 1,25 |
| `processed/mono` | ≈ 11–12 GB | số chính xác do `byte_premium.py` in ra |
| `build_mono` | ≈ 1–2 giờ/1 tiến trình | đo ≈ 2,5 MB/s; chạy song song theo ngôn ngữ để nhanh hơn |

Ngân sách mỗi ngôn ngữ = `(T/L) × bytes_per_token_en × prem_l`, lấy max qua các cấu hình lồng nhau (L2 ⊂ L4 ⊂ L8):
fra và zho theo L = 2; rus và tur theo L = 4; hin, arb, vie, eng theo L = 8. `_meta.json` của mỗi ngôn ngữ ghi ngân sách
của từng cấu hình để `study3/build_corpus.py` cắt lại. `bytes_per_token_en = 4,0` là **giả định**; chỉnh trong
`config.yaml` sau khi huấn luyện tokenizer và đo lại.

## Định dạng đầu ra

| Tệp | Mỗi dòng |
|---|---|
| `parallel/flores_{dev,devtest}.jsonl`, `ntrex.jsonl` | `{id, lang, text}`; cùng `id` = cùng nội dung ở mọi ngôn ngữ |
| `parallel/opus100_en-xx.jsonl` | như trên, hai dòng mỗi cặp (eng, xx) |
| `mono/{lang}/shard_*.jsonl` | `{id, lang, text}`, tài liệu đã NFKC, lọc trùng, lọc rò rỉ |
| `tasks/xnli/{train_eng,validation_L,test_L}.jsonl` | `{id, lang, idx, premise, hypothesis, label, label_name}` |
| `tasks/upos/{lang}.jsonl` | `{id, lang, split, sent_id, text, tokens, upos}` |
| `tasks/belebele/{lang}.jsonl` | `{id, lang, link, question_number, passage, question, answers[4], correct}` (200 câu, giống nhau ở mọi ngôn ngữ) |
| `typology/*.csv`, `wordfreq/{lang}.tsv` | ma trận khoảng cách / đặc trưng; `rank, word, count` |

Mã ngôn ngữ trong mọi tệp là mã 3 chữ của `config.yaml` (eng, fra, rus, hin, zho, vie, tur, arb, jpn, fin, swh, tha).

## Những điều cần biết (giới hạn và quyết định thiết kế)

- **Đã chạy thử thật trên mạng:** NTREX-128 (12 ngôn ngữ), UD (11 treebank, tag `r2.18`), WALS (`v2020.6`), `lang2vec`.
  **Chưa chạy thật** (môi trường thử không truy cập được): FLORES tarball, Hugging Face (XNLI, FineWeb-2, OPUS-100, GlotLID),
  Belebele zip, MUSE. Các đường code này được thử bằng dữ liệu/mô-đun giả; hãy chạy `--verify` và thử với `--scale 0.01`.
- **Tarball FLORES-200:** thử lần lượt `tinyurl.com/flores200dataset` và `dl.fbaipublicfiles.com/nllb/...`.
  Nếu cả hai hỏng: `--flores-source flores_plus` (cần HF_TOKEN, cấu trúc repo chưa kiểm chứng), hoặc tự đặt
  `raw/flores/{dev,devtest}/<mã>.<split>` (mỗi dòng một câu) rồi chạy tiếp `build_parallel.py`.
- **UD:** chỉ lấy `*-ud-{train,dev,test}.conllu` ở gốc repo (zip còn chứa tệp nguồn không thuộc bản phát hành; lấy nhầm sẽ
  làm EWT phình 33 000 câu thay vì 16 622). **Không có treebank Swahili** → `swh` không có UPOS. Thai-PUD chỉ có tập test (1 000 câu).
  Treebank tiếng Trung dùng `GSDSimp` (giản thể) cho khớp `zho_Hans`.
- **MUSE** chỉ có cặp en↔xx (không có Swahili). **XNLI** không có jpn, fin (đúng đặc tả). **FineWeb-2** dùng `cmn_Hani` cho tiếng Trung.
- **NFKC đổi dấu câu toàn góc của tiếng Trung/Nhật** (`，` → `,`). Đúng đặc tả, và áp dụng nhất quán cho cả dữ liệu đánh giá lẫn huấn luyện.
- **Lọc rò rỉ 13-gram** theo đơn vị: từ (chữ/số/dấu kết hợp, nên Hindi/Thái không bị cắt vụn), mỗi ký tự Hán/Kana/Thái là một đơn vị.
  Loại cả tài liệu nếu chứa ≥ 1 n-gram trùng. Câu đánh giá ngắn hơn 13 đơn vị không sinh n-gram nên **không** phát hiện được.
  Chỉ mục phải được xây **sau** `build_parallel.py` và `build_tasks.py`; thiếu nguồn nào, script cảnh báo bằng `WARNING`.
- **FineWeb-2 được stream, không tải toàn bộ:** lấy mẫu có xáo trộn cục bộ (seed cố định), đủ byte mục tiêu thì dừng. Không phải mẫu
  ngẫu nhiên đều theo thời gian của cả corpus. Tiếng Anh lấy từ FineWeb `sample-10BT`.
- **Tiếng Việt trong bảng tần suất:** đơn vị là âm tiết (tách theo khoảng trắng), không phải từ ghép. zho/jpn/tha cần
  `jieba`/`janome`/`pythainlp`, thiếu thì lùi về ký tự (ghi trong `wordfreq/<lang>.meta.json`).
- **WALS:** tiếng Ả Rập chuẩn (`ams`) chỉ có ~30 đặc trưng nên khoảng cách WALS của `arb` kém tin cậy (script cảnh báo);
  nên ưu tiên khoảng cách lang2vec. Khoảng cách lang2vec do `build_typology.py` tự tính (góc, như URIEL) vì lang2vec 1.1.x không còn hàm `distance`.
- **Eflomal** (Study 4) không nằm trong thư mục này; cài riêng khi tới Study 4.

## Xử lý sự cố

| Triệu chứng | Cách xử lý |
|---|---|
| `fetch.py xnli`: "không thấy parquet" | repo HF đổi cấu trúc; chạy `--verify`, xem `raw/xnli`, hoặc tải XNLI-1.0 thủ công |
| `datasets` báo lỗi script / `trust_remote_code` | `pip install -U datasets huggingface_hub` |
| `fetch.py fineweb2`: "Chưa có byte_premium.json" | chạy Bước 1 trước, hoặc `--bytes-per-lang N` |
| `build_mono`: `THIẾU dữ liệu` (shortfall) | tải thêm raw: tăng `raw_margin` trong `config.yaml`, xoá `raw/fineweb2/<lang>` rồi tải lại |
| `build_mono`: "Chưa có leakage_index.npz" | chạy `leakage_check.py build` (hoặc `--no-leakage` chỉ để thử) |
| Muốn làm lại một ngôn ngữ | `build_mono.py --langs fra --force`; `fetch.py fineweb2 --langs fra --force` |
| Manifest bị khoá treo | xoá `manifest.json.lock` |
