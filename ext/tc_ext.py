"""ThinkingCap worker extension (2026-10-26): sb_p5_ext.SBP5Ext (memguard + F1 + CAR + P5 hooks, as prod) plus the
Ornith ARX two-shot decode all-reduce (bit-identical to the live push all-reduce, half the PCIe wire bytes).
ARX is model-agnostic: it replaces CustomAllreduce.custom_all_reduce for admissible fp16 messages while a CUDA graph
is being captured. Enabled with SB_ORN_ARX=1 (SB_ORN_ARX_LIB / _MIN_BYTES / _MAX_BYTES / _MODE as in orn_arx.py).
The sum2 all-reduce (shared+routed expert partials) is a different op and stays on the live path."""
import importlib.abc
import importlib.util
import os
import sys

_ARX_PKG = os.environ.get("SB_ORN_ARX_PKG", "/opt/1cat/ornith122-trial/ports/arx")
_TARGET = "vllm.distributed.device_communicators.custom_all_reduce"


def _say(msg):
    print(f"TC-EXT pid {os.getpid()}: {msg}", flush=True)


class _AfterImport(importlib.abc.MetaPathFinder):
    def __init__(self, name, fn):
        self.name, self.fn, self.done = name, fn, False

    def find_spec(self, fullname, path, target=None):
        if fullname != self.name or self.done:
            return None
        self.done = True
        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            try:
                self.fn()
            except Exception as exc:  # noqa: BLE001
                _say(f"post-import hook for {fullname} failed: {exc!r}")

        loader.exec_module = exec_module
        return spec


if os.environ.get("SB_ORN_ARX", "0") == "1":
    if _ARX_PKG not in sys.path:
        sys.path.insert(0, _ARX_PKG)
    try:
        import orn_arx  # noqa: E402

        if _TARGET in sys.modules:
            orn_arx.install()
            _say("ARX installed (module already imported)")
        else:
            sys.meta_path.insert(0, _AfterImport(_TARGET, orn_arx.install))
            _say("ARX armed (post-import hook)")
    except Exception as exc:  # noqa: BLE001
        _say(f"ARX import FAILED {exc!r} (continuing without it)")

# TC-GLUE (2026-10-26): fused two-shot all-reduce + HC combine + grouped RMSNorm per hyper-connection boundary
# (ext/tc_glue.py, GLUE-README.md). TC_GLUE=1 enables; install() is lazy (post-import hooks) and never raises.
if os.environ.get("TC_GLUE", "0") == "1":
    try:
        import tc_glue  # noqa: E402

        tc_glue.install()
        _say(f"tc_glue {tc_glue.status()['status']}")
    except Exception as exc:  # noqa: BLE001
        _say(f"tc_glue import FAILED {exc!r} (continuing without it)")

from sb_p5_ext import SBP5Ext  # noqa: E402


class TCExt(SBP5Ext):
    def tc_glue_status(self):
        try:
            import tc_glue

            return tc_glue.status()
        except Exception as exc:  # noqa: BLE001
            return {"error": repr(exc)}

    def tc_arx_status(self):
        try:
            import orn_arx

            return dict(orn_arx._stats)
        except Exception as exc:  # noqa: BLE001
            return {"error": repr(exc)}
