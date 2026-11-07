"""ThinkingCap PROD worker extension (2026-10-26): whatever extension run-tc-best.sh chose (TC_BASE_EXT, default
tc_ext.TCExt = sb_p5_ext.SBP5Ext memguard/F1/CAR/P5 + optional ARX) PLUS the upstream PR #1232 backport
(tc_upfix_pr1232: the V2 model runner finishes RAM-cache stores of reused blocks before zeroing them). Needed whenever
KVOFF_GB > 0 (prod: 16 GiB RAM chat cache).

Use: --worker-extension-cls tc_prod_ext.TCProdExt   (run-tc-prod.sh sets WEXT=tc_prod_ext.TCProdExt and
TC_BASE_EXT=<the WEXT run-tc-best.sh would have used>). PYTHONPATH must contain /opt/1cat/tcflash-trial/ext
(run-tc-new.sh sets it). Env: TC_UPFIX_PR1232=0 disables the fix (A/B only). Nothing here is traced by torch.compile.
Log lines (serve log): "TC-EXT pid N: base ext ..." and "TC-EXT pid N: upfix PR#1232 installed=True"."""
import importlib
import os


def _say(msg):
    print(f"TC-EXT pid {os.getpid()}: {msg}", flush=True)


_BASE_NAME = os.environ.get("TC_BASE_EXT", "tc_ext.TCExt") or "tc_ext.TCExt"
if _BASE_NAME.startswith("tc_prod_ext."):   # never wrap ourselves
    _BASE_NAME = "tc_ext.TCExt"
_mod, _, _cls = _BASE_NAME.rpartition(".")
_Base = getattr(importlib.import_module(_mod), _cls)
_say(f"base ext {_BASE_NAME}")

import tc_upfix_pr1232 as _pr1232  # noqa: E402

try:
    _ok = _pr1232.install()
    _say(f"upfix PR#1232 installed={_ok}")
except Exception as exc:  # noqa: BLE001  never take the worker down for an optional fix; the gate checks the log line
    _say(f"upfix PR#1232 install FAILED {exc!r}")

import tc_sodraft_fix as _sodraft  # noqa: E402  JOB1 10-27: grammar requests see the drafts the worker verifies

try:
    _sd = _sodraft.install()
    _say(f"sodraft fix installed={_sd}")
except Exception as exc:  # noqa: BLE001
    _say(f"sodraft fix install FAILED {exc!r}")


class TCProdExt(_Base):
    def tc_prod_status(self):
        return {"base_ext": _BASE_NAME, "pr1232": _pr1232.status(), "sodraft": _sodraft.status()}
