"""Worker extension for TC_NOSYNC (staged 2026-10-30, NOT live). Wraps TC_NOSYNC_BASE_EXT (default tc_ext.TCExt, what
run-tc-best.sh uses) and imports tc_nosync (self-installs when TC_NOSYNC=1).
Prod chain: run-tc-prod.sh -> WEXT=tc_prod_ext.TCProdExt with TC_BASE_EXT=tc_nosync_ext.TCNoSyncExt -> tc_ext.TCExt.
Needs /opt/1cat/tcflash-trial/next-tps/ext on PYTHONPATH (run-tc-best.sh / run-tc-new.sh keep an inherited PYTHONPATH).
RPC: collective_rpc("tc_nosync_status")."""
import importlib
import os

_BASE_NAME = os.environ.get("TC_NOSYNC_BASE_EXT", "tc_ext.TCExt") or "tc_ext.TCExt"
if _BASE_NAME.startswith("tc_nosync_ext."):
    _BASE_NAME = "tc_ext.TCExt"
_mod, _, _cls = _BASE_NAME.rpartition(".")
_Base = getattr(importlib.import_module(_mod), _cls)
print(f"TC-NOSYNC pid {os.getpid()}: ext base {_BASE_NAME}", flush=True)

import tc_nosync as _ns  # noqa: E402


class TCNoSyncExt(_Base):
    def tc_nosync_status(self):
        return _ns.status()
