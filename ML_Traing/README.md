# Household Finance ML (`hfml`)

Pipeline Python cho hệ thống **AI tư vấn tài chính hộ gia đình**:
**5 bài toán Rule-Based**, **2 bài toán Machine Learning**, **LLM đóng vai giải thích kết quả**.

Kế hoạch chi tiết: [PLAN.md](PLAN.md) — 79 task, F01–F07, M01–M08, 8 tuần.

## Ranh giới cứng của hệ thống

| Tầng | Trách nhiệm | Không được làm |
|---|---|---|
| **Rule-Based** | Tính toán tài chính xác định | Không "học" gì cả |
| **Machine Learning** | Phân loại — trả về nhãn + xác suất | Không tự sinh khuyến nghị |
| **LLM** | Diễn đạt kết quả thành tiếng Việt | **Tuyệt đối không tính toán số** |

*AI là công cụ hỗ trợ phân tích và khuyến nghị tham khảo — không khẳng định đưa ra
giải pháp tài chính tối ưu, không quyết định thay người dùng.*

## Ràng buộc chi phối toàn bộ thiết kế

Hệ thống **chỉ thu thập dữ liệu một lần** qua form onboarding — không theo dõi giao dịch,
không có lịch sử nhiều tháng. Hệ quả:

- ❌ Không `lag`, `rolling`, xu hướng → **không làm dự báo chuỗi thời gian**
- ✅ Mọi mô hình là **cross-sectional** — một dòng hồ sơ vào, một kết quả ra
- ✅ Mọi feature tiền tệ là **tỉ lệ**, không phải giá trị tuyệt đối (xử lý domain gap VNĐ ↔ Home Credit)

## Chạy nhanh

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
pip install -e .                                     # cài package hfml

pytest -q                                            # kiểm tra bộ khung
```

Dataset Home Credit tải từ Kaggle vào `dataset/home-credit-default-risk/` — **không commit vào git**.

## Cấu trúc — 5 tầng theo luồng dữ liệu

```
Household-Finance-ML-Python/
├── config/
│   ├── config.yaml             # seed, ngưỡng tin cậy, tham số train, LLM
│   └── rules.yaml              # ngưỡng tầng rule + NGUỒN TRÍCH DẪN  (F02)
├── dataset/                    # Home Credit (không commit)
├── data/{raw,interim,processed}
├── docs/                       # dataset.md, model_card.md            (F07)
├── scripts/                    # entry-point chạy train, gọi vào src/hfml
├── src/training/runs/          # OUTPUT của mọi lần train:
│                               #   *.joblib + *.metadata.json
│                               #   results.csv — log mọi lần train    (F07)
│                               #   bảng so sánh, feature importance, biểu đồ
│                               #   (không chứa log — log ở logs/)
└── src/hfml/
    ├── config.py               # nạp cấu hình (env > yaml > default)
    ├── logger.py               # logging thống nhất
    │
    ├── data/                   # ── TẦNG 1 ─ F01 · M01 · Tuần 1
    │   ├── schema.py           #   data contract của form đầu vào
    │   ├── loader.py           #   nạp Home Credit
    │   ├── quality.py          #   kiểm tra chất lượng + hash dataset
    │   ├── synthetic.py        #   sinh dân số hộ cho ML01
    │   ├── preprocessing/      #   cleaner · encoders · pipeline
    │   └── features/           #   builder (feature tỉ lệ) · selection
    │
    ├── rules/                  # ── TẦNG 2 ─ F02 · M02 · Tuần 2
    │   ├── rb01_cashflow.py    #   thu, chi, số dư
    │   ├── rb02_health.py      #   sức khỏe tài chính → 4 mức
    │   ├── rb03_savings_goal.py#   tiến độ mục tiêu tiết kiệm
    │   ├── rb04_503020.py      #   phân bổ 50/30/20 đề xuất từ thu nhập
    │   ├── rb05_loan_capacity.py#  khả năng đáp ứng khoản vay
    │   ├── thresholds.py       #   nạp ngưỡng từ config/rules.yaml
    │   └── engine.py           #   chạy cả 5 rule
    │
    ├── ml/                     # ── TẦNG 3 ─ F03, F04 · M03, M04 · Tuần 3–4
    │   ├── base.py             #   contract: fit / predict / predict_proba
    │   ├── registry.py         #   lưu & tải artifact + metadata
    │   ├── ml01_recommendation/#   4 nhóm khuyến nghị (synthetic)
    │   ├── ml02_credit_risk/   #   rủi ro tín dụng Home Credit (nhãn thật)
    │   └── evaluation/         #   metrics · plots
    │
    ├── pipeline/               # ── TẦNG 4 ─ F05 · M05 · Tuần 5
    │   ├── normalizer.py       #   chuẩn hóa input inference
    │   ├── predictor.py        #   inference ML01 + ML02
    │   ├── result.py           #   schema structured result JSON
    │   ├── orchestrator.py     #   gom Rule + ML, kiểm tra confidence
    │   └── analyze.py          #   analyze(payload) -> Result
    │
    ├── llm/                    # ── TẦNG 5 ─ F05 · M06 · Tuần 6
    │   ├── prompts.py context.py client.py
    │   ├── guardrails.py validator.py
    │   └── chat.py
    │
    └── api/                    # ── vỏ giao tiếp ─ F05 · M08 · Tuần 8
        ├── schemas.py          #   Pydantic request/response
        └── main.py             #   POST /analyze · POST /chat · GET /health
```

Luồng chạy một hồ sơ:

```
form → data (validate + preprocess) → rules (RB01–RB05) ─┐
                                   → ml (ML01 + ML02) ───┴→ pipeline (structured JSON)
                                                             → llm (diễn đạt + validate số)
```

## Giao thức đánh giá

| | ML01 | ML02 |
|---|---|---|
| Nguồn nhãn | Synthetic, hàm `g(·)` tự thiết kế | `TARGET` — nhãn thật |
| Số lớp | 4 | 2 |
| Cân bằng | mỗi lớp ≥ 10% | 8,07% dương |
| Chia dữ liệu | `StratifiedKFold` | `StratifiedKFold` |
| **Chỉ số chọn model** | **Macro-F1** | **PR-AUC** |
| Baseline | `DummyClassifier(stratified)` | `DummyClassifier(stratified)` |
| Calibration | — | `CalibratedClassifierCV` + curve + Brier |
| Giải thích | Feature importance | Feature importance + SHAP (global + local) |
| Thuật toán | DT · Bagging · RF · XGBoost | DT · Bagging · RF · XGBoost |

Cùng `random_seed = 42`, cùng split, cùng feature set giữa 4 thuật toán —
nếu không thì bảng so sánh không có nghĩa.

**Accuracy không dùng để chọn model ở ML02**: đoán "không ai vỡ nợ" đã đạt 91,93%.
Accuracy vẫn báo cáo trong bảng, nhưng kết luận dựa trên PR-AUC.
