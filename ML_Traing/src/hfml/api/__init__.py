"""Tầng api — FastAPI demo (F05 · M08 · Tuần 8).

    schemas.py   task 16   Request/response schema bằng Pydantic
    main.py      task 17   POST /analyze, POST /chat
                 task 18   GET /health + load model một lần lúc startup

Phạm vi cố ý hẹp: FastAPI + database chỉ ở mức đủ demo (PLAN.md §1). API là
lớp vỏ mỏng gọi `hfml.pipeline.analyze()` — không chứa logic nghiệp vụ nào.

Đây là mục P2 trong thứ tự ưu tiên: nếu thiếu thời gian, thay bằng script
CLI demo (PLAN.md §13).
"""
