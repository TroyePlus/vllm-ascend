# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CANN ElasticBuffer offload for DeepSeek V4.1 Engram tables."""

import fcntl
import json
from contextlib import contextmanager
from enum import Enum
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch import nn
from vllm.logger import logger

E8M0_ONE_BITS = 127

_ELASTIC_BUFFER_FETCH_ABI = "auto"
_LEGACY_ENGRAM_FETCH_OP = None


@contextmanager
def _serialized_extension_build(name: str):
    """Serialize JIT builds and recover torch locks left by dead workers."""
    from torch.utils.cpp_extension import _get_build_directory

    build_directory = Path(_get_build_directory(name, verbose=False))
    build_directory.mkdir(parents=True, exist_ok=True)
    guard_path = build_directory / ".vllm_ascend_build.lock"
    torch_lock_path = build_directory / "lock"

    logger.info(
        "FOR-ENGRAM legacy ACLNN fetch bridge build lock waiting: path=%s",
        guard_path,
    )
    with guard_path.open("a+", encoding="utf-8") as guard:
        fcntl.flock(guard.fileno(), fcntl.LOCK_EX)
        try:
            logger.info(
                "FOR-ENGRAM legacy ACLNN fetch bridge build lock acquired: path=%s",
                guard_path,
            )
            # torch's cpp_extension lock is a create-only marker. If its owner
            # is killed it remains forever; the outer flock proves that no
            # current Engram builder in this deployment owns it.
            try:
                torch_lock_path.unlink()
            except FileNotFoundError:
                pass
            else:
                logger.warning(
                    "FOR-ENGRAM removed stale torch extension build lock: path=%s",
                    torch_lock_path,
                )
            yield build_directory
        finally:
            fcntl.flock(guard.fileno(), fcntl.LOCK_UN)


def _load_legacy_engram_fetch_op():
    """Build the inference-only bridge for the installed EngramFetch ABI."""
    global _LEGACY_ENGRAM_FETCH_OP
    if _LEGACY_ENGRAM_FETCH_OP is not None:
        return _LEGACY_ENGRAM_FETCH_OP

    from cann_ops_transformer.op_builder import OpBuilder

    class _LegacyEngramFetchOpBuilder(OpBuilder):
        def __init__(self):
            super().__init__("vllm_ascend_legacy_engram_fetch")

        def sources(self):
            return [str(Path(__file__).with_name("engram_fetch_legacy.cpp"))]

        def schema(self):
            return None

        def register_meta(self):
            return None

    builder = _LegacyEngramFetchOpBuilder()
    try:
        with _serialized_extension_build(builder.name) as build_directory:
            # The module cache is process-local. Recheck it after waiting in
            # case another thread in this worker completed the load.
            if _LEGACY_ENGRAM_FETCH_OP is not None:
                return _LEGACY_ENGRAM_FETCH_OP
            logger.info(
                "FOR-ENGRAM legacy ACLNN fetch bridge build started: directory=%s",
                build_directory,
            )
            _LEGACY_ENGRAM_FETCH_OP = builder.load(verbose=False)
    except Exception as exc:
        raise RuntimeError(
            "Failed to build the environment-compatible BF16 EngramFetch bridge"
        ) from exc
    logger.info("FOR-ENGRAM legacy ACLNN fetch bridge build completed")
    return _LEGACY_ENGRAM_FETCH_OP


class EngramTableState(Enum):
    EMPTY = "empty"
    STAGED = "staged"
    READY = "ready"
    POISONED = "poisoned"
    DESTROYED = "destroyed"


def create_engram_process_group(
    engram_tp_size: int,
    *,
    layer_id: int | None = None,
    expected_world_size: int | None = None,
):
    """Create a dedicated physical group for one Engram table."""
    if engram_tp_size <= 0:
        logger.error(
            "FOR-ENGRAM process group creation failed: layer=%s engram_tp_size=%d must be positive",
            layer_id,
            engram_tp_size,
        )
        raise ValueError("engram_tp_size must be positive")
    if not dist.is_available() or not dist.is_initialized():
        logger.error(
            "FOR-ENGRAM process group creation failed: layer=%s distributed is not initialized",
            layer_id,
        )
        raise RuntimeError("Engram ElasticBuffer requires an initialized distributed process group")

    world_size = dist.get_world_size()
    if expected_world_size is not None and world_size != expected_world_size:
        logger.error(
            "FOR-ENGRAM process group creation failed: layer=%s runtime_world_size=%d configured_world_size=%d",
            layer_id,
            world_size,
            expected_world_size,
        )
        raise RuntimeError(
            "Engram requires the PyTorch distributed world to span all configured "
            f"ranks; runtime={world_size}, configured={expected_world_size}"
        )
    if world_size % engram_tp_size:
        logger.error(
            "FOR-ENGRAM process group creation failed: layer=%s world_size=%d engram_tp_size=%d",
            layer_id,
            world_size,
            engram_tp_size,
        )
        raise ValueError(f"engram_tp_size={engram_tp_size} must divide distributed world_size={world_size}")
    rank = dist.get_rank()
    selected = None
    selected_ranks = None
    logger.info(
        "FOR-ENGRAM process group creation started: layer=%s rank=%d world_size=%d engram_tp_size=%d",
        layer_id,
        rank,
        world_size,
        engram_tp_size,
    )
    # Every rank creates all groups in the same order. Separate calls for the
    # Engram layers deliberately produce separate physical resources.
    try:
        for first in range(0, world_size, engram_tp_size):
            ranks = list(range(first, first + engram_tp_size))
            group = dist.new_group(ranks=ranks, backend=dist.get_backend())
            if rank in ranks:
                selected = group
                selected_ranks = ranks
    except BaseException:
        logger.exception(
            "FOR-ENGRAM process group creation failed: layer=%s rank=%d",
            layer_id,
            rank,
        )
        raise
    if selected is None:
        logger.error(
            "FOR-ENGRAM process group creation failed: no local group for layer=%s rank=%d",
            layer_id,
            rank,
        )
        raise RuntimeError("Failed to create the local Engram process group")
    logger.info(
        "FOR-ENGRAM process group creation completed: layer=%s rank=%d ranks=%s",
        layer_id,
        rank,
        selected_ranks,
    )
    return selected, True


class ElasticEngramEmbedding(nn.Module):
    """Row-sharded Engram table backed by CANN ElasticBuffer.

    The E4M3/E8M0 checkpoint shard is decoded on CPU before it is written as
    BF16. This matches the deployed ElasticBuffer ABI, which accepts one
    storage tensor and returns one BF16 tensor from Engram fetch.
    """

    def __init__(
        self,
        rows: int,
        width: int,
        engram_tp_size: int,
        group=None,
        owns_group: bool = False,
        layer_id: int | None = None,
        storage_format: str = "bf16",
        minimum_rows: int | None = None,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        if rows <= 0 or width <= 0:
            raise ValueError("Engram table dimensions must be positive")
        if engram_tp_size <= 0:
            raise ValueError("engram_tp_size must be positive")
        if storage_format != "bf16":
            raise ValueError("The current ElasticBuffer Engram ABI supports only bf16 storage")
        minimum_rows = rows if minimum_rows is None else int(minimum_rows)
        if minimum_rows <= 0 or minimum_rows > rows:
            raise ValueError("Engram minimum_rows must be positive and no greater than rows")
        if group is None:
            raise ValueError("ElasticBuffer Engram requires a non-null process group")

        self.rows = int(rows)
        self.width = int(width)
        self.minimum_rows = minimum_rows
        self.engram_tp_size = int(engram_tp_size)
        self.storage_format = storage_format
        self.group = group
        self.owns_group = owns_group
        self.layer_id = layer_id
        self.device = None if device is None else torch.device(device)
        self.rank = dist.get_rank(group)
        self.shard_rows = (self.rows + self.engram_tp_size - 1) // self.engram_tp_size
        self.padded_rows = self.shard_rows * self.engram_tp_size
        if self.padded_rows > torch.iinfo(torch.int32).max:
            raise ValueError("Engram global row IDs do not fit ElasticBuffer's int32 ABI")
        self.start = self.rank * self.shard_rows
        self.end = min(self.start + self.shard_rows, self.rows)
        self._host_weight: torch.Tensor | None = None
        self.engram_buffer = None
        self._fetch_in_flight = False
        self._state = EngramTableState.EMPTY
        logger.info(
            "FOR-ENGRAM table manager initialized: layer=%s tp_rank=%d "
            "tp_size=%d global_rows=%d minimum_rows=%d shard_rows=%d "
            "width=%d storage=%s",
            self.layer_id,
            self.rank,
            self.engram_tp_size,
            self.rows,
            self.minimum_rows,
            self.shard_rows,
            self.width,
            self.storage_format,
        )

    @property
    def state(self) -> EngramTableState:
        return self._state

    @staticmethod
    def _weight_map(root: Path) -> dict[str, str]:
        index_path = root / "model.safetensors.index.json"
        if index_path.is_file():
            return json.loads(index_path.read_text())["weight_map"]
        single_file = root / "model.safetensors"
        if not single_file.is_file():
            raise FileNotFoundError(f"No model.safetensors.index.json or model.safetensors under {root}")
        with safe_open(single_file, framework="pt", device="cpu") as file:
            return {key: single_file.name for key in file}

    @staticmethod
    def _scale_to_float(scale: torch.Tensor) -> torch.Tensor:
        if scale.dtype == torch.uint8:
            exponent = scale.to(torch.int32) - E8M0_ONE_BITS
            return torch.ldexp(
                torch.ones_like(exponent, dtype=torch.float32),
                exponent,
            )
        return scale.float()

    def _target_device(self) -> torch.device:
        if self.device is not None:
            return self.device
        npu = getattr(torch, "npu", None)
        if npu is None:
            raise RuntimeError("Engram distributed preflight requires an initialized NPU device")
        return torch.device("npu", npu.current_device())

    def _sync_phase_status(self, succeeded: bool, phase: str) -> bool:
        """Prevent entry into ElasticBuffer barriers after rank-local failure."""
        if self.engram_tp_size == 1:
            return succeeded
        status = torch.tensor(
            int(succeeded),
            dtype=torch.int32,
            device=self._target_device(),
        )
        dist.all_reduce(status, op=dist.ReduceOp.MIN, group=self.group)
        # This runs only during model startup, before any inference hot path.
        all_succeeded = bool(status.item())
        return all_succeeded

    def _release_staging(self) -> None:
        self._host_weight = None

    def _resolve_keys(self, root: Path, key: str) -> tuple[dict[str, str], str, str]:
        weight_map = self._weight_map(root)
        if key not in weight_map:
            alternate = key.removeprefix("model.") if key.startswith("model.") else f"model.{key}"
            if alternate in weight_map:
                key = alternate
        scale_key = key.removesuffix(".weight") + ".scale"
        if key not in weight_map or scale_key not in weight_map:
            raise KeyError(f"Engram checkpoint requires paired tensors {key} and {scale_key}")
        return weight_map, key, scale_key

    def _validate_checkpoint_metadata(
        self,
        key: str,
        weight_shape: list[int],
        weight_dtype: str,
        scale_key: str,
        scale_shape: list[int],
        scale_dtype: str,
    ) -> tuple[int, int]:
        if len(weight_shape) != 2 or weight_shape[1] != self.width:
            raise ValueError(f"{key}: expected [rows, {self.width}], got {weight_shape}")
        checkpoint_rows = weight_shape[0]
        if checkpoint_rows < self.minimum_rows or checkpoint_rows > self.rows:
            raise ValueError(
                f"{key}: checkpoint rows must be in [{self.minimum_rows}, {self.rows}], got {checkpoint_rows}"
            )
        if weight_dtype not in ("F8_E4M3", "F8_E4M3FN"):
            raise TypeError(f"{key}: expected FP8 E4M3 checkpoint data, got {weight_dtype}")
        if (
            len(scale_shape) != 2
            or scale_shape[0] != checkpoint_rows
            or scale_shape[1] <= 0
            or self.width % scale_shape[1]
        ):
            raise ValueError(f"{scale_key}: incompatible scale shape {scale_shape} for weight {weight_shape}")
        if scale_dtype not in ("F8_E8M0", "F8_E8M0FNU", "U8"):
            raise TypeError(f"{scale_key}: expected E8M0 or uint8 checkpoint data, got {scale_dtype}")
        block_size = self.width // scale_shape[1]
        return checkpoint_rows, block_size

    def _load_checkpoint_local(
        self,
        root: Path,
        key: str,
        chunk_rows: int,
    ) -> tuple[str, int, int, int]:
        if chunk_rows <= 0:
            raise ValueError("Engram chunk_rows must be positive")
        weight_map, key, scale_key = self._resolve_keys(root, key)
        with safe_open(root / weight_map[key], framework="pt", device="cpu") as weight_file:
            weight = weight_file.get_slice(key)
            weight_shape = weight.get_shape()
            weight_dtype = weight.get_dtype()
            with safe_open(root / weight_map[scale_key], framework="pt", device="cpu") as scale_file:
                scale = scale_file.get_slice(scale_key)
                scale_shape = scale.get_shape()
                checkpoint_rows, block_size = self._validate_checkpoint_metadata(
                    key,
                    weight_shape,
                    weight_dtype,
                    scale_key,
                    scale_shape,
                    scale.get_dtype(),
                )
                source_end = min(self.end, checkpoint_rows)
                copied_rows = max(source_end - self.start, 0)

                host_weight = torch.zeros(
                    (self.shard_rows, self.width),
                    dtype=torch.bfloat16,
                    device="cpu",
                )
                for start in range(self.start, source_end, chunk_rows):
                    stop = min(start + chunk_rows, source_end)
                    local_scale = self._scale_to_float(scale[start:stop])
                    valid_scale = torch.isfinite(local_scale) & (local_scale > 0)
                    if not bool(valid_scale.all()):
                        raise ValueError(f"{scale_key}: scales must be finite and positive")
                    decoded = (
                        weight[start:stop]
                        .float()
                        .unflatten(-1, (-1, block_size))
                        .mul_(local_scale.unsqueeze(-1))
                        .flatten(-2)
                    )
                    local_start = start - self.start
                    host_weight[local_start : local_start + stop - start].copy_(decoded)
                self._host_weight = host_weight.contiguous()
        return key, checkpoint_rows, copied_rows, block_size

    def load_checkpoint(
        self,
        model_path: str | Path,
        key: str,
        chunk_rows: int = 8192,
    ) -> None:
        if self._state is not EngramTableState.EMPTY:
            raise RuntimeError(f"Engram table {key} cannot load from state {self._state.value}")
        root = Path(model_path)
        logger.info(
            "FOR-ENGRAM checkpoint shard loading started: layer=%s key=%s tp_rank=%d row_range=[%d,%d) storage=%s",
            self.layer_id,
            key,
            self.rank,
            self.start,
            self.end,
            self.storage_format,
        )
        local_error: Exception | None = None
        result: tuple[str, int, int, int] | None = None
        try:
            result = self._load_checkpoint_local(root, key, chunk_rows)
        except Exception as exc:
            local_error = exc
            logger.exception(
                "FOR-ENGRAM checkpoint shard local preflight failed: layer=%s key=%s tp_rank=%d",
                self.layer_id,
                key,
                self.rank,
            )

        try:
            all_succeeded = self._sync_phase_status(local_error is None, "checkpoint-load")
        except Exception:
            self._release_staging()
            self._state = EngramTableState.POISONED
            logger.exception(
                "FOR-ENGRAM checkpoint distributed preflight failed: layer=%s tp_rank=%d",
                self.layer_id,
                self.rank,
            )
            raise
        if local_error is not None:
            self._release_staging()
            self._state = EngramTableState.POISONED
            raise local_error
        if not all_succeeded:
            self._release_staging()
            self._state = EngramTableState.POISONED
            raise RuntimeError(
                "Engram checkpoint loading failed on another rank; "
                "all ranks aborted before ElasticBuffer initialization"
            )

        assert result is not None
        resolved_key, checkpoint_rows, copied_rows, block_size = result
        self._state = EngramTableState.STAGED
        logger.info(
            "FOR-ENGRAM checkpoint shard loading completed: layer=%s key=%s "
            "tp_rank=%d checkpoint_rows=%d configured_rows=%d copied_rows=%d "
            "padded_shard_rows=%d checkpoint_scale_block_size=%d "
            "staged_dtype=%s state=%s",
            self.layer_id,
            resolved_key,
            self.rank,
            checkpoint_rows,
            self.rows,
            copied_rows,
            self.shard_rows,
            block_size,
            self._host_weight.dtype,
            self._state.value,
        )

    @staticmethod
    def _destroy_buffer_after_failure(buffer) -> None:
        if buffer is None:
            return
        try:
            buffer.destroy()
        except Exception:
            logger.exception("FOR-ENGRAM failed to destroy a partial ElasticBuffer")

    def offload_weights(self) -> None:
        """Write the staged BF16 shard into ElasticBuffer."""
        logger.info(
            "FOR-ENGRAM ElasticBuffer write function entered: layer=%s tp_rank=%d state=%s",
            self.layer_id,
            self.rank,
            self._state.value,
        )
        if self._state is EngramTableState.READY:
            return
        if self._state is not EngramTableState.STAGED:
            raise RuntimeError(
                f"Engram checkpoint shard must be staged before offload; current state is {self._state.value}"
            )
        assert self._host_weight is not None
        storage = self._host_weight
        weight_bytes = storage.numel() * storage.element_size()
        logger.info(
            "FOR-ENGRAM ElasticBuffer offload started: layer=%s tp_rank=%d "
            "shard_rows=%d width=%d storage=%s weight_bytes=%d",
            self.layer_id,
            self.rank,
            self.shard_rows,
            self.width,
            self.storage_format,
            weight_bytes,
        )

        local_error: Exception | None = None
        elastic_buffer_cls = None
        buffer = None
        num_cpu_bytes = 0
        try:
            from cann_ops_transformer.ops import ElasticBuffer as elastic_buffer_cls

            expected_weight_shape = (self.shard_rows, self.width)
            if not storage.is_cpu or not storage.is_contiguous() or tuple(storage.shape) != expected_weight_shape:
                raise RuntimeError(
                    "Engram staged weight must be a contiguous CPU tensor with "
                    f"shape {expected_weight_shape}, got {storage.device} "
                    f"{tuple(storage.shape)}"
                )
            if storage.dtype != torch.bfloat16:
                raise TypeError(f"BF16 Engram weight must be torch.bfloat16, got {storage.dtype}")
            num_cpu_bytes = elastic_buffer_cls.get_engram_storage_size_hint(
                self.shard_rows,
                self.width,
                storage.dtype,
            )
        except Exception as exc:
            local_error = exc
            logger.exception(
                "FOR-ENGRAM ElasticBuffer format preflight failed: layer=%s tp_rank=%d storage=%s",
                self.layer_id,
                self.rank,
                self.storage_format,
            )

        try:
            all_succeeded = self._sync_phase_status(local_error is None, "offload-format")
        except Exception:
            self._release_staging()
            self._state = EngramTableState.POISONED
            logger.exception(
                "FOR-ENGRAM ElasticBuffer format synchronization failed: layer=%s tp_rank=%d",
                self.layer_id,
                self.rank,
            )
            raise
        if local_error is not None or not all_succeeded:
            self._release_staging()
            self._state = EngramTableState.POISONED
            if local_error is not None:
                raise local_error
            raise RuntimeError(
                "Engram offload format preparation failed on another rank; "
                "all ranks aborted before ElasticBuffer initialization"
            )

        assert elastic_buffer_cls is not None
        local_error = None
        try:
            buffer = elastic_buffer_cls(
                self.group,
                num_cpu_bytes=num_cpu_bytes,
                explicitly_destroy=True,
            )
        except Exception as exc:
            local_error = exc
            logger.exception(
                "FOR-ENGRAM ElasticBuffer initialization failed: layer=%s tp_rank=%d storage=%s",
                self.layer_id,
                self.rank,
                self.storage_format,
            )
        try:
            all_succeeded = self._sync_phase_status(local_error is None, "elastic-buffer-init")
        except Exception:
            self._destroy_buffer_after_failure(buffer)
            self._release_staging()
            self._state = EngramTableState.POISONED
            logger.exception(
                "FOR-ENGRAM ElasticBuffer initialization synchronization failed: layer=%s tp_rank=%d",
                self.layer_id,
                self.rank,
            )
            raise
        if local_error is not None or not all_succeeded:
            self._destroy_buffer_after_failure(buffer)
            self._release_staging()
            self._state = EngramTableState.POISONED
            if local_error is not None:
                raise local_error
            raise RuntimeError(
                "Engram ElasticBuffer initialization failed on another rank; all ranks aborted before engram_write"
            )

        assert buffer is not None
        try:
            buffer.engram_write(storage)
        except Exception:
            self._destroy_buffer_after_failure(buffer)
            self._release_staging()
            self._state = EngramTableState.POISONED
            logger.exception(
                "FOR-ENGRAM ElasticBuffer write failed: layer=%s tp_rank=%d num_cpu_bytes=%d storage=%s",
                self.layer_id,
                self.rank,
                num_cpu_bytes,
                self.storage_format,
            )
            raise

        self.engram_buffer = buffer
        self._release_staging()
        self._state = EngramTableState.READY
        logger.info(
            "FOR-ENGRAM ElasticBuffer offload completed: layer=%s tp_rank=%d "
            "num_cpu_bytes=%d storage=%s staging_released=true state=%s",
            self.layer_id,
            self.rank,
            num_cpu_bytes,
            self.storage_format,
            self._state.value,
        )

    def _validate_fetch(
        self,
        output,
        expected_rows: int,
    ) -> torch.Tensor:
        expected_weight_shape = (expected_rows, self.width)
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("ElasticBuffer Engram fetch must return one BF16 tensor")
        if tuple(output.shape) != expected_weight_shape or output.dtype != torch.bfloat16:
            raise RuntimeError(
                f"ElasticBuffer returned {tuple(output.shape)} {output.dtype}, "
                f"expected {expected_weight_shape} BF16"
            )
        return output

    @staticmethod
    def _is_mixed_fetch_abi_error(exc: BaseException) -> bool:
        message = str(exc)
        return (
            "EngramFetch" in message
            and "Parameter fetched" in message
            and "must be 2D" in message
        )

    def _fetch_legacy_aclnn(self, flat_ids: torch.Tensor) -> torch.Tensor:
        """Call the BF16 inference ABI exposed by the installed EngramFetch."""
        buffer = self.engram_buffer
        assert buffer is not None
        context = getattr(buffer, "_engram_context_tensor", None)
        num_entries = getattr(buffer, "_engram_num_entries", None)
        if context is None or num_entries is None:
            raise RuntimeError(
                "Installed ElasticBuffer does not expose the state required "
                "by the environment-compatible EngramFetch bridge"
            )

        logger.info(
            "FOR-ENGRAM legacy ACLNN fetch parameters: layer=%s tp_rank=%d "
            "context_shape=%s indices_shape=%s indices_dtype=%s hidden_size=%d "
            "num_entries_per_rank=%d output_dtype=%s",
            self.layer_id,
            self.rank,
            tuple(context.shape),
            tuple(flat_ids.shape),
            flat_ids.dtype,
            self.width,
            int(num_entries),
            torch.bfloat16,
        )
        legacy_op = _load_legacy_engram_fetch_op()
        fetched = legacy_op.engram_fetch(
            context,
            flat_ids,
            self.width,
            int(num_entries),
        )
        result = torch.ops.cann_ops_transformer.engram_fetch_wait(context, fetched)
        return result

    def _fetch(self, flat_ids: torch.Tensor) -> torch.Tensor:
        global _ELASTIC_BUFFER_FETCH_ABI

        assert self.engram_buffer is not None
        if _ELASTIC_BUFFER_FETCH_ABI == "legacy-aclnn":
            return self._fetch_legacy_aclnn(flat_ids)

        try:
            wait_callable = self.engram_buffer.engram_fetch(flat_ids)
        except RuntimeError as exc:
            if not self._is_mixed_fetch_abi_error(exc):
                raise
            # The public wrapper marks the request in flight before entering
            # the operator. Tiling rejected it before launch, so it is safe to
            # clear the wrapper flag before using the compatibility call.
            if hasattr(self.engram_buffer, "_engram_fetch_in_progress"):
                self.engram_buffer._engram_fetch_in_progress = False
            logger.warning(
                "FOR-ENGRAM detected mixed ElasticBuffer ABI; switching to "
                "the environment-compatible legacy ACLNN bridge: layer=%s tp_rank=%d",
                self.layer_id,
                self.rank,
            )
            output = self._fetch_legacy_aclnn(flat_ids)
            _ELASTIC_BUFFER_FETCH_ABI = "legacy-aclnn"
        else:
            output = wait_callable()
            _ELASTIC_BUFFER_FETCH_ABI = "public"
        return output

    @torch.inference_mode()
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        logger.info(
            "FOR-ENGRAM ElasticBuffer fetch function entered: layer=%s tp_rank=%d ids_shape=%s state=%s",
            self.layer_id,
            self.rank,
            tuple(ids.shape),
            self._state.value,
        )
        if self._state is not EngramTableState.READY or self.engram_buffer is None:
            raise RuntimeError(f"Engram ElasticBuffer is not ready; current state is {self._state.value}")
        if self._fetch_in_flight:
            raise RuntimeError("Synchronous Engram does not permit overlapping fetches")
        shape = tuple(ids.shape)
        flat_ids = ids.reshape(-1).to(dtype=torch.int32).contiguous()
        self._fetch_in_flight = True
        try:
            output = self._fetch(flat_ids)
        except Exception:
            # ElasticBuffer only clears its internal in-flight flag after a
            # successful wait. Do not pretend the table can be reused.
            self._state = EngramTableState.POISONED
            logger.exception(
                "FOR-ENGRAM ElasticBuffer fetch failed and table was poisoned: layer=%s tp_rank=%d",
                self.layer_id,
                self.rank,
            )
            raise
        self._fetch_in_flight = False
        try:
            result = self._validate_fetch(output, flat_ids.numel())
        except Exception:
            self._state = EngramTableState.POISONED
            logger.exception(
                "FOR-ENGRAM ElasticBuffer fetch ABI validation failed: layer=%s tp_rank=%d storage=%s",
                self.layer_id,
                self.rank,
                self.storage_format,
            )
            raise
        return result.view(*shape, self.width)

    def destroy(self) -> None:
        if self._state is EngramTableState.DESTROYED:
            return
        if self._fetch_in_flight and self._state is not EngramTableState.POISONED:
            raise RuntimeError("Cannot destroy Engram while a fetch is in flight")
        logger.debug(
            "FOR-ENGRAM table destroy started: layer=%s tp_rank=%d buffer_initialized=%s owns_group=%s state=%s",
            self.layer_id,
            self.rank,
            self.engram_buffer is not None,
            self.owns_group,
            self._state.value,
        )
        try:
            if self.engram_buffer is not None:
                self.engram_buffer.destroy()
                self.engram_buffer = None
            if self.owns_group and self.group is not None and dist.is_initialized():
                dist.destroy_process_group(self.group)
        except BaseException:
            self._state = EngramTableState.POISONED
            logger.exception(
                "FOR-ENGRAM table destroy failed: layer=%s tp_rank=%d",
                self.layer_id,
                self.rank,
            )
            raise
        self.group = None
        self._release_staging()
        self._fetch_in_flight = False
        self._state = EngramTableState.DESTROYED
        logger.debug(
            "FOR-ENGRAM table destroy completed: layer=%s tp_rank=%d state=%s",
            self.layer_id,
            self.rank,
            self._state.value,
        )
