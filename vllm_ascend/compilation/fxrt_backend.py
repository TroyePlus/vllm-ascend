# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Route Dynamo-captured FX graphs to the external fxrt runtime."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any
from uuid import uuid4

from torch.fx import GraphModule

from vllm.logger import logger


def _safe_file_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return component or "model"


def dump_fx_graph(gm: GraphModule, dump_dir: Path, prefix: str) -> Path:
    """Persist a readable Dynamo FX graph before the backend is invoked."""
    dump_dir.mkdir(parents=True, exist_ok=True)
    name = _safe_file_component(prefix)
    path = dump_dir / f"fx_graph_{name}_pid{os.getpid()}_{uuid4().hex[:8]}.txt"
    path.write_text(
        f"{gm.print_readable(print_output=False)}\n\n"
        f"==== raw graph ====\n{gm.graph}\n",
        encoding="utf-8",
    )
    return path


def wrap_backend_with_fxrt(
    backend: str | Callable[..., Any],
    dump_dir: Path | None,
    prefix: str,
) -> Callable[..., Any]:
    """Wrap a backend so a string backend resolves to external fxrt.

    vLLM keeps ``inductor`` in its compilation config for validation, but the
    direct-fxrt path bypasses Triton Inductor entirely. ``dump_dir`` is optional
    so fxrt can also be used without writing graph dumps.
    """
    if isinstance(backend, str):
        from fxrt.torch import backend as compiler_fn
    else:
        compiler_fn = backend

    @wraps(compiler_fn)
    def fxrt_backend(
        gm: GraphModule, example_inputs: list[Any], **kwargs: Any
    ) -> Any:
        if dump_dir is not None:
            path = dump_fx_graph(gm, dump_dir, prefix)
            logger.info("Enter fxrt backend; Dynamo FX graph saved to %s", path)
        else:
            logger.info("Enter fxrt backend (FX graph dump disabled)")
        return compiler_fn(gm, example_inputs)

    return fxrt_backend
