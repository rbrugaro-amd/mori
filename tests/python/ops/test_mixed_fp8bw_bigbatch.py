# Large-batch repro of the offline blockwise-fp8 combine hang (fails >512 tok, offline hung at ~0.5 usage).
import mori, os, torch
os.environ.setdefault("MORI_ENABLE_SDMA", "1")
from tests.python.ops.dispatch_combine_test_utils import EpDispatchCombineTestCase
from tests.python.ops.test_dispatch_combine_async_ll import _make_asyncll_config

NTOK = int(os.environ.get("BIGBATCH_NTOK", "2048"))

def _test(rank, world_size, data_type, hidden_dim, max_tok, nexp_rank, nexp_tok):
    config = _make_asyncll_config(rank, world_size, data_type, hidden_dim, max_tok,
                                  nexp_rank, nexp_tok, quant_type="fp8_blockwise")
    op = mori.ops.EpDispatchCombineOp(config)
    op.enable_mixed_dispatch()
    tc = EpDispatchCombineTestCase(config)
    test_data = tc.gen_test_data(num_token_override=[NTOK]*world_size)
    (_, aidx, ain, aw, asc) = test_data
    r = config.rank
    do, dw, ds, di, drn = op.dispatch_send(ain[r], aw[r], asc[r], aidx[r])
    op.dispatch_recv(); tc.sync()
    co, _ = op.combine_send(do, None, aidx[r]); op.combine_recv(); tc.sync()
    tc.check_combine_result(op, test_data, co, None)

def test_bigbatch(torch_dist_process_manager):
    ws = 8
    for rank in range(ws):
        torch_dist_process_manager.task_queue.put((_test, [ws, torch.bfloat16, 7168, NTOK, 33, 9]))
    results = []
    for _ in range(ws):
        rank, res = torch_dist_process_manager.result_queue.get(); results.append((rank, res))
    for rank, res in sorted(results):
        print(f"[BIGBATCH ntok={NTOK}] rank {rank}: {'OK' if res is None else res}", flush=True)
    bad = [(r, x) for r, x in results if x is not None]
    assert not bad, f"{len(bad)} ranks failed"
