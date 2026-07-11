# Uneven per-rank batch repro (mimics DP-attention imbalance incl edge counts 0/1).
import mori, os, torch
os.environ.setdefault("MORI_ENABLE_SDMA", "1")
from tests.python.ops.dispatch_combine_test_utils import EpDispatchCombineTestCase
from tests.python.ops.test_dispatch_combine_async_ll import _make_asyncll_config

# uneven counts incl 0, 1, odd, and large near-max
COUNTS = [0, 1, 3, 777, 1023, 1536, 2047, 2048]
MAXTOK = 2048

def _test(rank, world_size, data_type, hidden_dim, max_tok, nexp_rank, nexp_tok):
    config = _make_asyncll_config(rank, world_size, data_type, hidden_dim, max_tok,
                                  nexp_rank, nexp_tok, quant_type="fp8_blockwise")
    op = mori.ops.EpDispatchCombineOp(config)
    op.enable_mixed_dispatch()
    tc = EpDispatchCombineTestCase(config)
    test_data = tc.gen_test_data(num_token_override=COUNTS)
    (_, aidx, ain, aw, asc) = test_data
    r = config.rank
    do, dw, ds, di, drn = op.dispatch_send(ain[r], aw[r], asc[r], aidx[r])
    op.dispatch_recv(); tc.sync()
    co, _ = op.combine_send(do, None, aidx[r]); op.combine_recv(); tc.sync()
    tc.check_combine_result(op, test_data, co, None)

def test_uneven(torch_dist_process_manager):
    ws = 8
    for rank in range(ws):
        torch_dist_process_manager.task_queue.put((_test, [ws, torch.bfloat16, 7168, MAXTOK, 33, 9]))
    results = []
    for _ in range(ws):
        rank, res = torch_dist_process_manager.result_queue.get(); results.append((rank, res))
    for rank, res in sorted(results):
        print(f"[UNEVEN counts={COUNTS}] rank {rank}: {'OK' if res is None else str(res)[:200]}", flush=True)
    bad = [(r, x) for r, x in results if x is not None]
    assert not bad, f"{len(bad)} ranks failed"
