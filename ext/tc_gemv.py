"""tc_gemv - runtime hook that routes ThinkingCap's remaining small-M dense FP16 projections to an SM70 GEMV kernel.

ThinkingCap = Qwen3.8 Flash-Next NVFP4 on /opt/1cat/vllm-main-20261025, TP4 on 4x V100. 2026-10-26.
Default OFF. Enabled with TC_GEMV=1 (see GEMV-README.md). Nothing here edits engine files; everything is monkey-patched
in the worker process from a post-import hook (same pattern as tc_ext.py / sb_p5_ext.py).

What the engine does today (sm70_fp16_gemv.py): the FP16 GEMV lane admits only M=1 (`_runtime_ok`), and M>1 is served
by hand-written kernels for a fixed set of shapes (router 512x2560 M<=4 + MTP rows%5==0, out/o_proj 2560x1536 M<=8,
GDN input, HC, shared gate_up rows%5==0). Everything else at M=2..20 (MTP verify width 5 per user, drafts M=users)
goes to torch.nn.functional.linear -> cuBLAS/cutlass:
    .self_attn.qkv_proj            (3584, 2560)  13x/round (12 QSA layers + MTP layer)   ~41 us each at M=5
    .self_attn.indexer.index_qk_proj (640, 2560) 13x/round                               ~18 us
    .mlp.shared_expert.down_proj   (2560, 160)   49x/round (48 layers + MTP layer)       ~9 us
    .mlp.shared_expert_gate        (1, 2560)     49x/round  (cuBLAS dot + reduce, 2 launches) ~7.5 us
    .linear_attn.out_proj/.self_attn.o_proj (2560,1536) 49x/round only when M > 8 (3-4 users)
    final hyper_connection_mixer (nn.Linear 320x10240 / 10240x320), MTP fc_embedding/fc_hidden (640x2560)
The two shared-expert layers are not even in the lane's role table (so their M=1 draft calls also go to cuBLAS).

Three hooks, all opt-in:
  1. role table: `_ROLE_PLANS` gets `.mlp.shared_expert.down_proj` and `.mlp.shared_expert_gate`, so those layers
     receive Qwen38SM70FP16LinearMethod (engine's Triton row GEMV at M=1; hook 2 at M>1).           TC_GEMV_ROLES_EXTEND=1
  2. Qwen38SM70FP16LinearMethod.apply -> torch.ops.tc_gemv_py.proj (opaque op): admitted (role, M) -> tc_gemv kernel,
     everything else -> the engine's own op unchanged (torch.ops.vllm.qwen38_sm70_fp16_gemv).       TC_GEMV=1
  3. (optional) final HC mixer F.linear calls in common/hyperconnection.py and MTP fc_* via UnquantizedLinearMethod.apply.
                                                                                   TC_GEMV_MIXER=1 / TC_GEMV_FC=1
Admission is inside the opaque op (never a Python branch on M in the traced graph), exactly like the engine's lane, so
prefill, M>20, non-FP16 and non-contiguous inputs keep today's path. Packed-only layers keep their own shortcut.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO = os.environ.get("TC_GEMV_LIB", os.path.join(_HERE, "build", "tc_gemv_C.so"))
_GEMV_MOD = "vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv"
_LINEAR_MOD = "vllm.model_executor.layers.linear"
_HC_COMMON_MOD = "vllm.models.qwen4_exp.common.hyperconnection"

ENABLED = os.environ.get("TC_GEMV", "0") == "1"
MIN_M = int(os.environ.get("TC_GEMV_MIN_M", "2"))
MAX_M = min(20, int(os.environ.get("TC_GEMV_MAX_M", "20")))
ROLES_EXTEND = os.environ.get("TC_GEMV_ROLES_EXTEND", "1") == "1"
MIXER = os.environ.get("TC_GEMV_MIXER", "0") == "1"
FC = os.environ.get("TC_GEMV_FC", "0") == "1"
VERBOSE = os.environ.get("TC_GEMV_VERBOSE", "1") == "1"

# role suffix -> (allowed M range) ; "engine" means: only when the engine's own M>1 kernel declines (mirrors its gate)
_DEFAULT_ROLES = {
    ".self_attn.qkv_proj": (MIN_M, MAX_M),
    ".self_attn.indexer.index_qk_proj": (MIN_M, MAX_M),
    ".mlp.shared_expert.down_proj": (MIN_M, MAX_M),
    ".mlp.shared_expert_gate": (MIN_M, MAX_M),
    ".linear_attn.out_proj": "engine",
    ".self_attn.o_proj": "engine",
}


def _parse_roles(spec: str | None) -> dict:
    """TC_GEMV_ROLES="suffix[:lo-hi],..." overrides the default table (empty value keeps defaults)."""
    if not spec:
        return dict(_DEFAULT_ROLES)
    roles = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            name, rng = item.split(":", 1)
            if rng == "engine":
                roles[name] = "engine"
            else:
                lo, hi = rng.split("-")
                roles[name] = (int(lo), min(20, int(hi)))
        else:
            roles[item] = (MIN_M, MAX_M)
    return roles


ROLES = _parse_roles(os.environ.get("TC_GEMV_ROLES"))

# Extra role-table rows handed to the engine's lane (shape per TP4 rank, plan = (block_k, num_warps, load_policy)).
# The row kernel masks K tails, so block_k 256 for K=160 and the router plan for the 1-row gate.
ROLE_TABLE_EXTENSION = (
    (".mlp.shared_expert.down_proj", (2560, 160), (256, 2, 0)),
    (".mlp.shared_expert_gate", (1, 2560), (1024, 8, 0)),
)

# Static inventory (per TP4 rank, 1 user) of every dense FP16 projection in the decode round; "route" = best stack
# today (CT_TM bridge, MTP_RB/MTP_SB on, hybrid PLE), from the mp0 profile + sm70_fp16_gemv.py gating.
INVENTORY = (
    # role, shape (N,K), calls/round at 1u, M values (verify / draft), route today, hooked by tc_gemv
    ("attn/mlp HC input_mix_weight_down_block_inject", (336, 10240), 98, "5 / 1", "HC batch kernels (packed-only)", "no"),
    ("attn/mlp HC input_mix_weight_up", (10240, 320), 98, "5 / 1", "HC batch kernels (packed-only)", "no"),
    ("linear_attn.in_proj_qkvz + in_proj_ba", (4096 + 24, 2560), 36, "5 / -", "gdn_input_batch (packed-only)", "no"),
    ("linear_attn.out_proj", (2560, 1536), 36, "5 / -", "dense_batch_kernel<1536> (M<=8) else cuBLAS", "M>8"),
    ("self_attn.qkv_proj", (3584, 2560), 13, "5 / 1", "cuBLAS cutlass 16x16 splitK8 (M=5) / Triton row GEMV (M=1)", "M>=2"),
    ("self_attn.indexer.index_qk_proj", (640, 2560), 13, "5 / 1", "cuBLAS cutlass 32x32 splitK4 + reduce / Triton (M=1)", "M>=2"),
    ("self_attn.o_proj", (2560, 1536), 13, "5 / 1", "dense_batch_kernel<1536> (M<=8) / Triton (M=1)", "M>8"),
    ("mlp.gate (router)", (512, 2560), 49, "5 / 1", "router_batch (rows%5==0) / dense_batch (M<=4) / Triton (M=1)", "no"),
    ("mlp.shared_expert.gate_up_proj", (320, 2560), 49, "5 / 1", "shared_up_batch (rows%5==0) / cuBLAS gemv (M=1)", "no (own op)"),
    ("mlp.shared_expert.down_proj", (2560, 160), 49, "5 / 1", "cuBLAS gemmSN (M=5) / cuBLAS gemv (M=1)", "M>=2 (+table M=1)"),
    ("mlp.shared_expert_gate", (1, 2560), 49, "5 / 1", "cuBLAS dot+reduce 2 launches (M=5) / cuBLAS (M=1)", "M>=2 (+table M=1)"),
    ("ple.key_proj / value_proj (layer 2)", (10240, 2560), 1, "5", "cuBLAS (eager, once per verify)", "no"),
    ("hyper_connection_mixer input_mix_weight_down (nn.Linear)", (320, 10240), 2, "5 / 1", "cuBLAS cutlass splitK20", "TC_GEMV_MIXER=1"),
    ("hyper_connection_mixer input_mix_weight_up (nn.Linear)", (10240, 320), 3, "5 / 1", "cuBLAS cutlass 16x16", "TC_GEMV_MIXER=1"),
    ("mtp fc_embedding / fc_hidden", (640, 2560), 2, "5 and 20 / 1 and 4", "cuBLAS s884 splitK10 / gemv batched", "TC_GEMV_FC=1"),
    ("lm_head (verify, target)", (62080, 2560), 1, "5", "cuBLAS cutlass 16x16 splitK8, 483 us (660 GB/s)", "no"),
    ("lm_head (drafts)", (62080, 2560), 4, "1", "channel-QPN8 FP8 head (MTPQPN8Head)", "no"),
)

_lock = threading.Lock()
_state: dict = {"installed": False, "lib": False, "hits": {}, "announced": set(), "log": []}


def _say(msg: str) -> None:
    line = f"TC-GEMV pid {os.getpid()}: {msg}"
    _state["log"].append(line)
    if VERBOSE:
        print(line, file=sys.stderr, flush=True)


def load_lib() -> bool:
    if _state["lib"]:
        return True
    import torch

    if not os.path.exists(_SO):
        _say(f"kernel library missing: {_SO} (run ext/build/build.py)")
        return False
    torch.ops.load_library(_SO)
    _state["lib"] = True
    return True


# ---------------------------------------------------------------- admission (runtime, inside the opaque op)
def _role_range(role: str):
    for suffix, rng in ROLES.items():
        if role.endswith(suffix):
            return rng
    return None


def admit(x, weight, role: str, *, engine_declines=None, require_cuda: bool = True) -> bool:
    """True when (x, weight, role) should run on the tc_gemv kernel. Pure shape/dtype/role logic; `engine_declines`
    is a callable returning True when the engine's own M>1 kernel would NOT take this call ("engine" roles)."""
    import torch

    if not ENABLED or (require_cuda and not _state["lib"]):
        return False
    rng = _role_range(role)
    if rng is None or x.ndim != 2 or weight.ndim != 2:
        return False
    m, k = int(x.shape[0]), int(x.shape[1])
    n = int(weight.shape[0])
    if rng == "engine":
        if not (2 <= m <= MAX_M) or engine_declines is None or not engine_declines():
            return False
    else:
        lo, hi = rng
        if not (lo <= m <= hi):
            return False
    if (
        int(weight.shape[1]) != k
        or k % 32 != 0
        or n < 1
        or x.dtype != torch.float16
        or weight.dtype != torch.float16
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or x.data_ptr() % 16 != 0
        or weight.data_ptr() % 16 != 0
    ):
        return False
    if require_cuda and not (x.is_cuda and weight.is_cuda and x.device == weight.device):
        return False
    return True


def _run_kernel(x, weight, role: str):
    import torch

    out = x.new_empty((x.shape[0], weight.shape[0]))
    torch.ops.tc_gemv.gemv_out(out, x, weight, 0)
    key = (".".join(role.split(".")[-2:]), int(x.shape[0]))
    if key not in _state["announced"]:
        _state["announced"].add(key)
        _say(f"active: {key[0]} M={key[1]} shape={tuple(weight.shape)}")
    return out


# ---------------------------------------------------------------- opaque op + hooks
def _register_ops():
    import torch
    from torch.library import Library

    if "proj" in _state:
        return
    lib = Library("tc_gemv_py", "DEF")
    lib.define("proj(Tensor x, Tensor weight, str role, Tensor? packed_router, bool dense_batch) -> Tensor")
    lib.define("linear(Tensor x, Tensor weight, str role) -> Tensor")

    def proj(x, weight, role, packed_router, dense_batch):
        g = sys.modules[_GEMV_MOD]

        def engine_declines():
            # out/o_proj: the engine's exact dense batch takes M<=8 (with its reduced-precision carve-out)
            return not (dense_batch and g._can_use_dense_batch(x, weight, role))

        if admit(x, weight, role, engine_declines=engine_declines):
            return _run_kernel(x, weight, role)
        return torch.ops.vllm.qwen38_sm70_fp16_gemv(x, weight, role, packed_router, dense_batch)

    def proj_fake(x, weight, role, packed_router, dense_batch):
        return x.new_empty((*x.shape[:-1], weight.shape[0]))

    def linear(x, weight, role):
        x2 = x.reshape(-1, x.shape[-1])
        if admit(x2, weight, role):
            return _run_kernel(x2, weight, role).reshape(*x.shape[:-1], weight.shape[0])
        return torch.nn.functional.linear(x, weight)

    def linear_fake(x, weight, role):
        return x.new_empty((*x.shape[:-1], weight.shape[0]))

    lib.impl("proj", proj, "CUDA")
    lib.impl("linear", linear, "CUDA")
    lib._register_fake("proj", proj_fake)
    lib._register_fake("linear", linear_fake)
    _state["proj_lib"] = lib
    _state["proj"] = proj
    _state["linear"] = linear


def _hook_gemv_module(g) -> None:
    """Patch the engine's lane module (role table + apply)."""
    import torch

    if ROLES_EXTEND:
        have = {s for s, _, _ in g._ROLE_PLANS}
        extra = tuple(
            (s, shape, g._GemvPlan(*plan)) for s, shape, plan in ROLE_TABLE_EXTENSION if s not in have
        )
        if extra:
            g._ROLE_PLANS = tuple(g._ROLE_PLANS) + extra
            g._SHAPE_PLANS = {shape: plan for _, shape, plan in g._ROLE_PLANS}
            _say(f"role table +{len(extra)}: {[s for s, _, _ in extra]}")
    cls = g.Qwen38SM70FP16LinearMethod
    if getattr(cls.apply, "_tc_gemv_orig", None) is not None:
        return
    orig_apply = cls.apply
    decode = g.use_sm70_decode_graph_semantics

    def tc_apply(self, layer, x, bias=None):
        if (
            bias is not None
            or getattr(layer, "_sm70_packed_only_kind", None)
            or not decode()
            or getattr(layer, "weight", None) is None
            or not isinstance(layer.weight, torch.Tensor)
        ):
            return orig_apply(self, layer, x, bias)
        return torch.ops.tc_gemv_py.proj(
            x,
            layer.weight,
            str(getattr(layer, "prefix", "")),
            getattr(layer, "_sm70_mtp_router_packed", None),
            bool(getattr(layer, "_sm70_qwen38_dense_batch", False)),
        )

    tc_apply._tc_gemv_orig = orig_apply
    cls.apply = tc_apply
    _state["installed"] = True
    _say(f"Qwen38SM70FP16LinearMethod.apply hooked; roles={sorted(ROLES)} M={MIN_M}..{MAX_M}")


def _hook_linear_module(lm) -> None:
    """Optional: MTP fc_embedding/fc_hidden stay on UnquantizedLinearMethod (sm70_mtp_fc requires that exact type)."""
    import torch

    if not FC:
        return
    cls = lm.UnquantizedLinearMethod
    if getattr(cls.apply, "_tc_gemv_orig", None) is not None:
        return
    orig_apply = cls.apply

    def tc_apply(self, layer, x, bias=None):
        prefix = str(getattr(layer, "prefix", ""))
        if bias is None and prefix.endswith((".fc_embedding", ".fc_hidden")) and x.dtype == torch.float16:
            return torch.ops.tc_gemv_py.linear(x, layer.weight, prefix)
        return orig_apply(self, layer, x, bias)

    tc_apply._tc_gemv_orig = orig_apply
    cls.apply = tc_apply
    ROLES.setdefault(".fc_embedding", (MIN_M, MAX_M))
    ROLES.setdefault(".fc_hidden", (MIN_M, MAX_M))
    _say("UnquantizedLinearMethod.apply hooked for .fc_embedding/.fc_hidden")


def _hook_hc_common_module(hc) -> None:
    """Optional: the final hyper_connection_mixer uses raw nn.Linear weights through this module's `F.linear`."""
    import torch

    if not MIXER:
        return
    real_F = hc.F
    if getattr(real_F, "_tc_gemv_shim", False):
        return

    class _FShim:
        _tc_gemv_shim = True

        def __getattr__(self, name):
            return getattr(real_F, name)

        @staticmethod
        def linear(x, weight, bias=None):
            if bias is None and x.dtype == torch.float16 and weight.ndim == 2 and weight.shape[1] % 32 == 0:
                role = ".hyper_connection_mixer.mix" if weight.shape[0] == 320 else ".hyper_connection_mixer.up"
                return torch.ops.tc_gemv_py.linear(x, weight, role)
            return real_F.linear(x, weight, bias)

    hc.F = _FShim()
    ROLES.setdefault(".hyper_connection_mixer.mix", (MIN_M, MAX_M))
    ROLES.setdefault(".hyper_connection_mixer.up", (MIN_M, MAX_M))
    _say("common/hyperconnection F.linear shimmed for the final mixer (TC_GEMV_MIXER=1)")


class _AfterImport(importlib.abc.MetaPathFinder):
    def __init__(self, targets: dict):
        self.pending = dict(targets)

    def find_spec(self, fullname, path, target=None):
        fn = self.pending.pop(fullname, None)
        if fn is None:
            return None
        if not self.pending and self in sys.meta_path:
            sys.meta_path.remove(self)
        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            return spec
        real_exec = spec.loader.exec_module

        def exec_module(module):
            real_exec(module)
            try:
                fn(module)
            except Exception as exc:  # noqa: BLE001
                _say(f"hook for {fullname} FAILED {exc!r} (engine path unchanged)")

        spec.loader.exec_module = exec_module
        return spec


def install() -> str:
    """Arm the hooks (idempotent). Called at import when TC_GEMV=1."""
    with _lock:
        if _state.get("armed"):
            return status()
        if not ENABLED:
            return "disabled (TC_GEMV!=1)"
        if not load_lib():
            return "disabled (library missing)"
        _register_ops()
        targets = {_GEMV_MOD: _hook_gemv_module, _LINEAR_MOD: _hook_linear_module, _HC_COMMON_MOD: _hook_hc_common_module}
        pending = {}
        for name, fn in targets.items():
            if name in sys.modules:
                fn(sys.modules[name])
            else:
                pending[name] = fn
        if pending:
            sys.meta_path.insert(0, _AfterImport(pending))
        _state["armed"] = True
        _say(f"armed (pending imports: {sorted(pending)})")
        return status()


def status() -> str:
    return (
        f"tc_gemv enabled={ENABLED} lib={_state['lib']} installed={_state['installed']} "
        f"M={MIN_M}..{MAX_M} roles={sorted(ROLES)} active={sorted(_state['announced'])}"
    )


def print_inventory() -> None:
    print(f"{'projection':62s} {'(N,K)/rank':>14s} {'calls':>5s}  {'M':12s} {'route today':60s} hooked")
    for role, shape, calls, m, route, hooked in INVENTORY:
        print(f"{role:62s} {str(shape):>14s} {calls:5d}  {m:12s} {route:60s} {hooked}")


if __name__ == "__main__":
    if "--inventory" in sys.argv:
        print_inventory()
    else:
        print(status())
elif ENABLED:
    install()
