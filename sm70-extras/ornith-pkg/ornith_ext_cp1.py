"""ornith_ext + upstream PR #1232 backport (2026-10-25). Worker extension: --worker-extension-cls ornith_ext_cp1.OrnithExt
Imports ornith_ext unchanged (all its runtime patches), then installs orn_upfix_pr1232 (V2 runner finishes RAM-cache
stores of reused blocks before zeroing them)."""
import ornith_ext as _base
from ornith_ext import *  # noqa: F401,F403
from ornith_ext import OrnithExt  # noqa: F401

import orn_upfix_pr1232 as _pr1232

try:
    _ok = _pr1232.install()
    _base._say(f"upfix PR#1232 installed={_ok}")
except Exception as exc:   # never take the worker down for an optional fix; the gate checks the log line
    _base._say(f"upfix PR#1232 install FAILED {exc!r}")
