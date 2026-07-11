# Mixed split dispatch/combine round-trip gate: IntraNode dispatch + AsyncLL combine.
import mori
import os
import torch
os.environ.setdefault("MORI_ENABLE_SDMA", "1")
from tests.python.ops.dispatch_combine_test_utils import (
    EpDispatchCombineTestCase,
    assert_worker_results,
)
from tests.python.ops.test_dispatch_combine_async_ll import _make_asyncll_config


def _test_mixed(rank, world_size, data_type, hidden_dim,
                max_num_inp_token_per_rank, num_experts_per_rank, num_experts_per_token):
    config = _make_asyncll_config(
        rank, world_size, data_type, hidden_dim,
        max_num_inp_token_per_rank, num_experts_per_rank, num_experts_per_token,
    )
    op = mori.ops.EpDispatchCombineOp(config)
    op.enable_mixed_dispatch()  # IntraNode dispatch + AsyncLL combine
    tc = EpDispatchCombineTestCase(config)
    test_data = tc.gen_test_data()
    (_, all_rank_indices, all_rank_input, all_rank_weights, all_rank_scales) = test_data
    r = config.rank
    dispatch_output, dispatch_weights, dispatch_scales, dispatch_indices, dispatch_recv_num_token = op.dispatch_send(
        all_rank_input[r], all_rank_weights[r], all_rank_scales[r], all_rank_indices[r]
    )
    op.dispatch_recv()  # mirror the model's two-phase dispatch (no-op in mixed)
    tc.sync()
    # Round-trip: AsyncLL two-phase combine (send+recv) of the unchanged dispatched tokens.
    combine_output, _ = op.combine_send(dispatch_output, None, all_rank_indices[r])
    op.combine_recv()
    tc.sync()
    tc.check_combine_result(op, test_data, combine_output, None)


def test_mixed_dispatch_combine(torch_dist_process_manager):
    world_size = 8
    for rank in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_test_mixed, [world_size, torch.bfloat16, 7168, 128, 33, 9])
        )
    results = []
    for _ in range(world_size):
        rank, result = torch_dist_process_manager.result_queue.get()
        results.append((rank, result))
    for rank, result in sorted(results, key=lambda x: x[0]):
        print(f"[MIXED-RT] rank {rank}: {'OK' if result is None else result}", flush=True)
    bad = [(r, x) for r, x in results if x is not None]
    assert not bad, f"{len(bad)} ranks failed"
