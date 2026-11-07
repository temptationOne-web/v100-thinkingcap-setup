"""Worker extension = tc_ext.TCExt (SBP5Ext + ARX) + the tc_gemv small-M dense GEMV hook. 2026-10-26.
Select with WEXT=tc_gemv_ext.TCGemvExt (run-tc-best.sh honours WEXT from the environment; tc_ext.py is untouched).
The hook itself is armed only when TC_GEMV=1 (tc_gemv.py), so this class is inert otherwise."""
import tc_gemv  # noqa: F401  (arms the post-import hooks when TC_GEMV=1)
from tc_ext import TCExt


class TCGemvExt(TCExt):
    def tc_gemv_status(self):
        return tc_gemv.status()
