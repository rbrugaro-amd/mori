# Force cross-node path on a single node by shrinking gpu_per_node -> exercises the early-firing
# per-token RDMA path (dests on the "other" logical node). Also reveals if per-token RDMA floods.
import mori, os, torch
os.environ.setdefault("MORI_ENABLE_SDMA", "1")
from tests.python.ops.dispatch_combine_test_utils import EpDispatchCombineTestCase

GPN = int(os.environ.get("XNODE_GPN", "4"))

def _cfg(rank, ws, gpn):
    return mori.ops.EpDispatchCombineConfig(
        data_type=torch.bfloat16, rank=rank, world_size=ws, hidden_dim=7168,
        scale_dim=0, scale_type_size=1, max_num_inp_token_per_rank=2048,
        num_experts_per_rank=33, num_experts_per_token=9, max_token_type_size=4,
        block_num=64, warp_num_per_block=8,
        kernel_type=mori.ops.EpDispatchCombineKernelType.AsyncLL,
        max_total_recv_tokens=0, quant_type="fp8_blockwise", gpu_per_node=gpn)

def _test(rank, ws, gpn):
    config = _cfg(rank, ws, gpn)
    op = mori.ops.EpDispatchCombineOp(config); op.enable_mixed_dispatch()
    tc = EpDispatchCombineTestCase(config)
    data = tc.gen_test_data(num_token_override=[2048]*ws)
    _, aidx, ain, aw, asc = data; r = config.rank
    do, dw, ds, di, drn = op.dispatch_send(ain[r], aw[r], asc[r], aidx[r]); op.dispatch_recv(); tc.sync()
    co, _ = op.combine_send(do, None, aidx[r]); op.combine_recv(); tc.sync()
    tc.check_combine_result(op, data, co, None)

def test_xnode(torch_dist_process_manager):
    ws = 8
    for rank in range(ws):
        torch_dist_process_manager.task_queue.put((_test, [ws, GPN]))
    results = []
    for _ in range(ws):
        rank, res = torch_dist_process_manager.result_queue.get(); results.append((rank, res))
    for rank, res in sorted(results):
        print(f"[XNODE gpn={GPN}] rank {rank}: {'OK' if res is None else str(res)[:250]}", flush=True)
    bad = [(r, x) for r, x in results if x is not None]
    assert not bad, f"{len(bad)} ranks failed"
