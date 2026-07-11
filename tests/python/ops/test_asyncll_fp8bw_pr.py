import torch
from tests.python.ops.test_dispatch_combine_async_ll import _test_dispatch_combine
def _run(qt, mgr):
    ws = 8
    for _ in range(ws):
        mgr.task_queue.put((_test_dispatch_combine, [ws, torch.bfloat16, 7168, 128, 32, 8, 0, 1, qt]))
    res = [mgr.result_queue.get() for _ in range(ws)]
    for rank, r in sorted(res):
        print(f"[ASYNCLL {qt}] rank {rank}: {'OK' if r is None else str(r)[:200]}", flush=True)
    return [(x, y) for x, y in res if y is not None]
def test_none(torch_dist_process_manager):
    assert not _run("none", torch_dist_process_manager)
def test_fp8_blockwise(torch_dist_process_manager):
    assert not _run("fp8_blockwise", torch_dist_process_manager)
