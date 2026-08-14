"""Chuỗi báo cáo ML01: chấm test → so sánh → importance → vẽ hình.

Gom bốn bước của F03 task 11–13 vào một lời gọi, để nơi nào cần báo cáo cũng
chạy đúng cùng một trình tự: `scripts/report_ml01.py` khi chạy riêng, và cuối
mỗi `scripts/train_*.py` khi train xong.

Vì sao là module chứ không nằm trong script
--------------------------------------------
Hai chỗ cùng cần chuỗi này. Chép đôi thì đến lúc thêm một bước, một bên được
sửa còn bên kia lặng lẽ sinh báo cáo thiếu — mà báo cáo thiếu không báo lỗi,
nó chỉ ra số cũ.

Lưu ý về tập test
-----------------
`evaluate_on_test()` chấm tập test mỗi lần chạy. Việc này KHÔNG quay ngược
ảnh hưởng training hay chọn model — `select_final_model()` (task 14) chỉ đọc
chỉ số CV — nhưng người đọc số vẫn nên nhớ rằng tập test đã được nhìn nhiều
lần, nên nó không còn là ước lượng "hoàn toàn chưa thấy" như lần chấm đầu.
Con số đem báo cáo cuối cùng nên lấy từ một lần chạy sạch trên seed đã chốt.
"""
from __future__ import annotations

from hfml.data.synthetic import PopulationParams
from hfml.logger import get_logger
from hfml.ml.evaluation.plots import generate_ml01_plots, plot_results_table
from hfml.ml.ml01_recommendation.train import (
    compare_models,
    evaluate_on_test,
    feature_importance_report,
)

log = get_logger(__name__)


def build_ml01_report(
    params: PopulationParams | None = None,
    seed: int | None = None,
    n_splits: int | None = None,
    top_n: int = 10,
    runs_dir=None,
) -> dict:
    """Chạy cả bốn bước, trả `dict` gồm bảng số và đường dẫn các hình.

    Trình tự bắt buộc: ba bước đầu ghi CSV, bước vẽ đọc lại chính các CSV đó.
    Đảo thứ tự thì hàm vẽ ném `FileNotFoundError` — cố ý, để hình không bao
    giờ được dựng từ số của lần chạy trước.
    """
    log.info("── Task 11: chấm 4 thuật toán trên tập test ──")
    evaluation = evaluate_on_test(params, seed=seed, runs_dir=runs_dir)

    log.info("── Task 12: so sánh CV với test ──")
    comparison = compare_models(params, seed=seed, n_splits=n_splits,
                                runs_dir=runs_dir)

    log.info("── Task 13: độ quan trọng feature ──")
    importance = feature_importance_report(params, seed=seed, top_n=top_n,
                                           runs_dir=runs_dir)

    log.info("── Vẽ hình ──")
    figures = generate_ml01_plots(runs_dir=runs_dir, top_n=top_n)
    figures["results_table"] = plot_results_table(runs_dir=runs_dir)

    return {
        "test_summary": evaluation["summary"],
        "comparison": comparison["comparison"],
        "importance_pivot": importance["pivot"],
        "figures": figures,
    }
