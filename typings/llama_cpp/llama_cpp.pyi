# SPDX-License-Identifier: MIT
from collections.abc import Callable

class _PerfContextData:
    n_p_eval: int
    n_eval: int
    t_p_eval_ms: float
    t_eval_ms: float

# A ctypes CFUNCTYPE: wraps a Python callable, or makes a null pointer when
# called with no argument.
ggml_abort_callback: Callable[..., object]

def llama_perf_context(ctx: object, /) -> _PerfContextData: ...
def llama_set_abort_callback(ctx: object, abort_callback: object, user_data: object, /) -> None: ...
