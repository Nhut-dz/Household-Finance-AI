"""Chốt phiên bản dataset và sinh tài liệu (F01 task 6 + 7).

    python scripts/build_dataset_docs.py            # băm SHA-256 đầy đủ (~10s)
    python scripts/build_dataset_docs.py --no-hash  # bỏ băm, chạy nhanh khi thử
    python scripts/build_dataset_docs.py --verify   # chỉ so với manifest đã chốt

Sinh ra hai file, cả hai ĐỀU commit vào git:

    docs/dataset_manifest.json   SHA-256 + kích thước + phân bố nhãn
    docs/dataset.md              tài liệu đọc được cho báo cáo

Chạy `--verify` trước mỗi lần train lại: dữ liệu đổi mà metric cũng đổi thì
biết ngay nguyên nhân nằm ở dữ liệu chứ không phải code.
"""
from __future__ import annotations

import argparse
import sys

from hfml.data import loader, quality
from hfml.logger import get_logger, log_run_context

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-hash", action="store_true",
                        help="bỏ qua SHA-256 (nhanh, nhưng không chốt được phiên bản)")
    parser.add_argument("--verify", action="store_true",
                        help="so dataset hiện tại với manifest đã chốt rồi thoát")
    parser.add_argument("--skip-quality", action="store_true",
                        help="bỏ qua kiểm tra chất lượng (đỡ đọc hết 122 cột)")
    args = parser.parse_args()
    compute_hash = not args.no_hash

    log_run_context(log)

    # Chỉ chặn khi thiếu file BẮT BUỘC. Trước đây chặn theo `available_files()`
    # nên vắng `previous_application.csv` — file ngoài phạm vi, cố ý không giữ —
    # cũng làm script từ chối chạy và không sinh lại được manifest.
    missing = loader.missing_required()
    if missing:
        log.error("Thiếu file dataset bắt buộc: %s", ", ".join(missing))
        log.error("Tải từ %s", quality.KAGGLE_URL)
        return 1

    absent = [n for n in loader.OPTIONAL_FILES if not loader.resolve(n).exists()]
    if absent:
        log.info("Không có (ngoài phạm vi, không cần thiết): %s", ", ".join(absent))

    if args.verify:
        drift = quality.verify_manifest(compute_hash=compute_hash)
        if drift:
            log.error("Dataset LỆCH so với manifest:")
            for line in drift:
                log.error("  - %s", line)
            return 1
        log.info("Dataset khớp manifest.")
        return 0

    manifest = quality.build_manifest(compute_hash=compute_hash)
    quality.write_manifest(manifest)

    report = None
    if not args.skip_quality:
        report = quality.check_application_train()
        log.info("Chất lượng: %d vấn đề (%d lỗi)",
                 len(report.issues), len(report.by_severity(quality.Severity.ERROR)))
        for issue in report.issues:
            log.info("  %s", issue)

    quality.write_dataset_doc(manifest, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
