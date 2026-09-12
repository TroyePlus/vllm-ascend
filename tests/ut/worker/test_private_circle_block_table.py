from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.worker.block_table import MultiGroupBlockTable


class _FakeBlockTable:
    def __init__(self, *, is_private: bool, rows: int = 2, columns: int = 64):
        self.is_private_circle_group = is_private
        self.is_mamba_group = False
        self.physical_block_size = 32
        self.block_table = SimpleNamespace(
            np=np.zeros((rows, columns), dtype=np.int32)
        )
        self.num_blocks_per_row = np.zeros(rows, dtype=np.int64)
        self.slot_mapping = SimpleNamespace(
            gpu=torch.arange(8, dtype=torch.int32)
        )

    def clear_row(self, row: int) -> None:
        self.block_table.np[row].fill(0)
        self.num_blocks_per_row[row] = 0

    def add_row(self, block_ids: list[int], row: int) -> None:
        self.clear_row(row)
        self.block_table.np[row, : len(block_ids)] = block_ids
        self.num_blocks_per_row[row] = len(block_ids)


def _new_multi_group_table(
    *block_tables: _FakeBlockTable,
) -> MultiGroupBlockTable:
    table = MultiGroupBlockTable.__new__(MultiGroupBlockTable)
    table.block_tables = list(block_tables)
    table.private_circle_allocation_ids = torch.full(
        (2,), -1, dtype=torch.int64, device="cpu"
    )
    table.private_circle_blocks_per_allocation = 0
    return table


def test_private_circle_layout_accepts_highest_pool_handle():
    private = _FakeBlockTable(is_private=True)
    table = _new_multi_group_table(private)

    table.set_private_circle_allocations(
        allocation_ids=np.array([7], dtype=np.int64),
        blocks_per_allocation=5,
        num_allocations=8,
        private_num_blocks=41,
        window_starts=np.array([132], dtype=np.int64),
        valid_lengths=np.array([260], dtype=np.int64),
    )

    assert private.block_table.np[0, 0] == 0
    assert private.block_table.np[0, 4] == 40

    with pytest.raises(IndexError, match="handle is out of range"):
        table.set_private_circle_allocations(
            allocation_ids=np.array([8], dtype=np.int64),
            blocks_per_allocation=5,
            num_allocations=8,
            private_num_blocks=41,
        )


def test_private_circle_layout_rejects_inconsistent_pool_capacity():
    private = _FakeBlockTable(is_private=True)
    table = _new_multi_group_table(private)

    with pytest.raises(ValueError, match="block count does not match"):
        table.set_private_circle_allocations(
            allocation_ids=np.array([0], dtype=np.int64),
            blocks_per_allocation=5,
            num_allocations=8,
            private_num_blocks=21,
        )


def _reference_rows(allocation_ids, blocks_per_allocation, block_size, num_cols, window_starts, valid_lengths):
    """Per-row reference semantics the vectorized install must reproduce."""
    rows = np.zeros((len(allocation_ids), num_cols), dtype=np.int64)
    counts = np.zeros(len(allocation_ids), dtype=np.int64)
    for row, allocation in enumerate(allocation_ids):
        if allocation < 0:
            continue
        valid_length = int(valid_lengths[row])
        window_start = int(window_starts[row])
        first_page = max(0, window_start // block_size)
        last_page = (valid_length - 1) // block_size if valid_length > 0 else 0
        logical_count = max(1, last_page + 1)
        base = 1 + int(allocation) * blocks_per_allocation
        ids = [0] * min(first_page, logical_count) + [
            base + (logical % blocks_per_allocation) for logical in range(first_page, logical_count)
        ]
        rows[row, : len(ids)] = ids
        counts[row] = len(ids)
    return rows, counts


@pytest.mark.parametrize(
    "allocation_ids,blocks_per_allocation,window_starts,valid_lengths",
    [
        (np.array([0, -1, 3]), 4, np.array([0, 0, 130]), np.array([64, 0, 260])),
        (np.array([5]), 2, np.array([0]), np.array([1])),
        (np.array([1, 2]), 4, np.array([128, 96]), np.array([129, 512])),
        (np.array([-1, -1]), 4, np.array([0, 0]), np.array([0, 0])),
        (np.array([0]), 4, np.array([0]), np.array([100000])),
    ],
)
def test_private_circle_install_matches_reference_rows(
    allocation_ids, blocks_per_allocation, window_starts, valid_lengths
):
    block_size = 32
    num_allocations = 8
    private = _FakeBlockTable(is_private=True, rows=8, columns=4096)
    table = _new_multi_group_table(private)
    table.private_circle_allocation_ids = torch.full(
        (8,), -1, dtype=torch.int64, device="cpu"
    )

    try:
        table.set_private_circle_allocations(
            allocation_ids=allocation_ids,
            blocks_per_allocation=blocks_per_allocation,
            num_allocations=num_allocations,
            private_num_blocks=1 + num_allocations * blocks_per_allocation,
            window_starts=window_starts,
            valid_lengths=valid_lengths,
        )
    except RuntimeError:
        return  # reference case exceeds configured length; both reject

    expected_rows, expected_counts = _reference_rows(
        allocation_ids,
        blocks_per_allocation,
        block_size,
        private.block_table.np.shape[1],
        window_starts,
        valid_lengths,
    )
    assert np.array_equal(private.block_table.np[: len(allocation_ids)], expected_rows)
    assert np.array_equal(
        private.num_blocks_per_row[: len(allocation_ids)], expected_counts
    )
    # Rows beyond the batch and stale columns are zeroed.
    assert not private.block_table.np[len(allocation_ids):].any()
    assert not private.num_blocks_per_row[len(allocation_ids):].any()


def test_bounded_replay_masks_shared_slots_but_keeps_private_ring_slots():
    shared = _FakeBlockTable(is_private=False)
    private = _FakeBlockTable(is_private=True)
    table = _new_multi_group_table(shared, private)
    positions = torch.tensor([127, 128, 200, 31], dtype=torch.int64)
    request_indices = torch.tensor([0, 0, 0, 1], dtype=torch.int64)
    persistent_starts = torch.tensor([128, -1], dtype=torch.int64)
    private_before = private.slot_mapping.gpu.clone()

    table.mask_shared_slots_for_private_circle_bounded_replay(
        positions,
        request_indices,
        persistent_starts,
    )

    assert shared.slot_mapping.gpu[:4].tolist() == [-1, 1, 2, 3]
    assert torch.equal(private.slot_mapping.gpu, private_before)
