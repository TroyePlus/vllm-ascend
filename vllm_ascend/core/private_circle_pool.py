# SPDX-License-Identifier: Apache-2.0
"""Request-private circle allocation bookkeeping."""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from vllm.logger import logger


PRIVATE_CIRCLE_TRANSFER_NONE = "none"
PRIVATE_CIRCLE_TRANSFER_RECEIVING = "receiving"
PRIVATE_CIRCLE_TRANSFER_READY = "ready"
PRIVATE_CIRCLE_TRANSFER_FAILED = "failed"


def is_private_circle_kv_cache_spec(spec: object) -> bool:
    """Return whether a leaf KV cache spec is a V4.1 private-circle SWA spec."""
    from vllm_ascend.core.deepseek_v41 import DeepseekV41SWASpec

    return isinstance(spec, DeepseekV41SWASpec)


def compute_prefix_bounded_replay_start(
    hit_length: int,
    bounded_replay_tokens: int,
) -> int:
    """Return the exact request-private circle recompute start.

    Shared cache accounting stays at ``hit_length``. The returned position is
    only used to prepend private-circle warm-up tokens in the model runner, so it
    must not be aligned to a shared-cache block size.
    """
    if hit_length < 0 or bounded_replay_tokens < 0:
        raise ValueError("hit_length and bounded_replay_tokens must be non-negative")
    return max(0, hit_length - bounded_replay_tokens)


def detect_new_prefix_hit_length(
    computed_before_schedule: int,
    computed_after_schedule: int,
    num_scheduled_tokens: int,
    waiting_for_remote_kvs: bool = False,
) -> int | None:
    """Detect a first-step prefix hit from scheduler-visible positions.

    Local cache hits are normally recorded at ``get_computed_blocks``. This
    fallback covers external KV connectors, including asynchronous loads that
    advance ``num_computed_tokens`` without scheduling model tokens.
    """
    if min(
        computed_before_schedule,
        computed_after_schedule,
        num_scheduled_tokens,
    ) < 0:
        raise ValueError("scheduler token counts must be non-negative")
    if computed_before_schedule > 0 and not waiting_for_remote_kvs:
        return None
    confirmed_length = max(0, computed_after_schedule - num_scheduled_tokens)
    return confirmed_length or None


def compute_private_circle_prefill_workspace_blocks(
    max_num_batched_tokens: int,
    max_num_seqs: int,
    window_size: int,
    block_size: int,
) -> int:
    """Return the worst-case shared Prefill workspace size in blocks.

    A request needs either up to ``window_size - 1`` restored history tokens
    or up to ``window_size`` bounded replay tokens. Bounded replay rows restore no history,
    so the larger transient range is reserved once, plus page-alignment slack
    for every request.
    """
    if max_num_batched_tokens < 0:
        raise ValueError("max_num_batched_tokens must be non-negative")
    if max_num_seqs <= 0 or window_size <= 0 or block_size <= 0:
        raise ValueError(
            "max_num_seqs, window_size, and block_size must be positive"
        )
    max_workspace_tokens = (
        max_num_batched_tokens
        + max_num_seqs * (window_size + block_size - 1)
    )
    return (max_workspace_tokens + block_size - 1) // block_size


@dataclass(frozen=True)
class PrivateCircleConfig:
    """Sizing parameters for the request-private circle pool.

    Compact ring sizing for A5. Repeated physical IDs are masked at token/offset granularity.
    """

    block_size: int
    window_size: int
    in_flight_tokens: int
    max_num_seqs: int
    delayed_free_slots: int | None = None

    @property
    def capacity_tokens(self) -> int:
        return self.window_size - 1 + self.in_flight_tokens

    @property
    def blocks_per_allocation(self) -> int:
        capacity = self.capacity_tokens
        return (capacity + self.block_size - 1) // self.block_size

    @property
    def num_allocations(self) -> int:
        delayed = self.max_num_seqs if self.delayed_free_slots is None else self.delayed_free_slots
        return self.max_num_seqs + delayed

    @property
    def num_blocks(self) -> int:
        return 1 + self.num_allocations * self.blocks_per_allocation




class PrivateCirclePool:
    """Small, thread-safe allocation ledger for private circle slots.

    Physical KV tensors are owned by the KV-cache allocator; this class only
    owns request-to-allocation bookkeeping and is intentionally independent of
    prefix-cache block hashes.
    """

    def __init__(self, config: PrivateCircleConfig):
        if (config.block_size <= 0
                or config.window_size <= 0
                or config.in_flight_tokens <= 0
                or config.max_num_seqs <= 0):
            raise ValueError("private circle sizing values must be positive")
        self.config = config
        if config.delayed_free_slots is not None and config.delayed_free_slots < 0:
            raise ValueError("delayed_free_slots must be non-negative")
        self._free = list(range(config.num_allocations))
        self._owned: dict[str, int] = {}
        # Key retained entries by allocation, not request. A request id may
        # be admitted again while a previous P/D transfer still owns its old
        # allocation; request-keyed storage would overwrite and leak it.
        self._retained: dict[int, str] = {}
        # Allocations detached by the deferred-free path, not yet returned.
        self._detached: set[int] = set()
        self._refs: dict[int, int] = {}
        self._lock = Lock()
        logger.info(
            "PRIVATE_CIRCLE_POOL init block_size=%d window_size=%d in_flight=%d "
            "blocks_per_allocation=%d allocations=%d total_blocks=%d",
            config.block_size,
            config.window_size,
            config.in_flight_tokens,
            config.blocks_per_allocation,
            config.num_allocations,
            config.num_blocks,
        )

    def reserve(self, request_id: str) -> int | None:
        with self._lock:
            if request_id in self._owned:
                allocation = self._owned[request_id]
                logger.debug(
                    "PRIVATE_CIRCLE_POOL reserve_reuse request_id=%s allocation=%d "
                    "blocks=%d-%d free_allocations=%d",
                    request_id,
                    allocation,
                    self._first_block_id(allocation),
                    self._last_block_id(allocation),
                    len(self._free),
                )
                return allocation
            if not self._free:
                logger.debug(
                    "PRIVATE_CIRCLE_POOL reserve_exhausted request_id=%s "
                    "allocations=%d retained=%d",
                    request_id,
                    self.config.num_allocations,
                    len(self._retained),
                )
                return None
            allocation = self._free.pop()
            self._owned[request_id] = allocation
            logger.debug(
                "PRIVATE_CIRCLE_POOL reserve request_id=%s allocation=%d "
                "blocks=%d-%d free_allocations=%d",
                request_id,
                allocation,
                self._first_block_id(allocation),
                self._last_block_id(allocation),
                len(self._free),
            )
            return allocation

    def release(self, request_id: str) -> bool:
        with self._lock:
            allocation = self._owned.pop(request_id, None)
            if allocation is None:
                logger.debug("PRIVATE_CIRCLE_POOL release_missing request_id=%s", request_id)
                return False
            self._return_allocation_locked(allocation, request_id)
            return True

    def detach(self, request_id: str) -> int | None:
        """Drop ownership without returning the slot to the free list.

        Mirrors pop_blocks_for_free: bookkeeping drops now (a preempted
        request reserves a fresh slot on resume), the physical return is
        deferred until the owning step's fence completes.
        """
        with self._lock:
            allocation = self._owned.pop(request_id, None)
            if allocation is not None:
                self._detached.add(allocation)
                logger.debug(
                    "PRIVATE_CIRCLE_POOL detach request_id=%s allocation=%d "
                    "blocks=%d-%d refs=%d",
                    request_id,
                    allocation,
                    self._first_block_id(allocation),
                    self._last_block_id(allocation),
                    self._refs.get(allocation, 0),
                )
            return allocation

    def return_detached_allocation(self, allocation: int, request_id: str) -> None:
        """Return a detached allocation to the free list (ref-aware,
        idempotent: a repeated drain must not double-free a slot)."""
        with self._lock:
            self._validate_allocation(allocation)
            if allocation not in self._detached:
                logger.debug(
                    "PRIVATE_CIRCLE_POOL return_detached_missing allocation=%d",
                    allocation,
                )
                return
            self._detached.discard(allocation)
            self._return_allocation_locked(allocation, request_id)

    def _return_allocation_locked(self, allocation: int, request_id: str) -> None:
        refs = self._refs.get(allocation, 0)
        retained = refs > 0
        if retained:
            self._retained[allocation] = request_id
        else:
            self._free.append(allocation)
        logger.debug(
            "PRIVATE_CIRCLE_POOL release request_id=%s allocation=%d "
            "blocks=%d-%d refs=%d retained=%s free_allocations=%d",
            request_id,
            allocation,
            self._first_block_id(allocation),
            self._last_block_id(allocation),
            refs,
            retained,
            len(self._free),
        )

    def acquire_ref(self, allocation: int) -> None:
        with self._lock:
            self._validate_allocation(allocation)
            refs = self._refs.get(allocation, 0) + 1
            self._refs[allocation] = refs
            logger.debug(
                "PRIVATE_CIRCLE_POOL acquire_ref allocation=%d blocks=%d-%d refs=%d",
                allocation,
                self._first_block_id(allocation),
                self._last_block_id(allocation),
                refs,
            )

    def release_ref(self, allocation: int) -> bool:
        with self._lock:
            self._validate_allocation(allocation)
            refs = self._refs.get(allocation, 0)
            if refs <= 0:
                logger.debug(
                    "PRIVATE_CIRCLE_POOL release_ref_missing allocation=%d", allocation
                )
                return False
            released_to_free = False
            if refs == 1:
                self._refs.pop(allocation, None)
                if allocation in self._retained:
                    self._retained.pop(allocation)
                    self._free.append(allocation)
                    released_to_free = True
                refs = 0
            else:
                refs -= 1
                self._refs[allocation] = refs
            logger.debug(
                "PRIVATE_CIRCLE_POOL release_ref allocation=%d blocks=%d-%d refs=%d "
                "released_to_free=%s free_allocations=%d",
                allocation,
                self._first_block_id(allocation),
                self._last_block_id(allocation),
                refs,
                released_to_free,
                len(self._free),
            )
            return True

    def _first_block_id(self, allocation: int) -> int:
        return 1 + allocation * self.config.blocks_per_allocation

    def _last_block_id(self, allocation: int) -> int:
        return self._first_block_id(allocation) + self.config.blocks_per_allocation - 1

    def allocation_block_ids(self, allocation: int) -> range:
        self._validate_allocation(allocation)
        start = self._first_block_id(allocation)
        return range(start, start + self.config.blocks_per_allocation)

    def slot_for_position(self, allocation: int, absolute_position: int) -> int:
        """Return the physical slot for an absolute token position."""
        if absolute_position < 0:
            raise ValueError("absolute position must be non-negative")
        block_ids = self.allocation_block_ids(allocation)
        ring_block = (absolute_position // self.config.block_size) % len(block_ids)
        return block_ids[ring_block] * self.config.block_size + absolute_position % self.config.block_size

    def get_allocation(self, request_id: str) -> int | None:
        with self._lock:
            return self._owned.get(request_id)

    def request_for_allocation(self, allocation: int) -> str | None:
        """Return the live or retained owner of an allocation."""
        with self._lock:
            for request_id, owned in self._owned.items():
                if owned == allocation:
                    return request_id
            return self._retained.get(allocation)

    def ref_count(self, allocation: int) -> int:
        with self._lock:
            self._validate_allocation(allocation)
            return self._refs.get(allocation, 0)

    def contains(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._owned

    @property
    def free_allocations(self) -> int:
        with self._lock:
            return len(self._free)

    @property
    def retained_allocations(self) -> int:
        with self._lock:
            return len(self._retained)

    @property
    def used_allocations(self) -> int:
        with self._lock:
            return self.config.num_allocations - len(self._free)

    def over_high_watermark(self, threshold: float = 0.9) -> bool:
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        return (self.used_allocations
                / self.config.num_allocations
                >= threshold)

    def _validate_allocation(self, allocation: int) -> None:
        if allocation < 0 or allocation >= self.config.num_allocations:
            raise IndexError("invalid private circle allocation")
