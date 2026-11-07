"""vLLM general plugin: segment-cached prompt tokenization in the API server.  STAGED 2026-10-30, NOT INSTALLED.

Patches vllm.utils.async_utils.AsyncMicrobatchTokenizer.encode (the call the renderer uses for every chat prompt,
kwargs = add_special_tokens=False, truncation=True, max_length=max_input_tokens+1). Only long string prompts (> 20k
chars) with exactly those kwargs take the cached path; everything else, and any prompt whose ids would exceed
max_length (the truncation case), goes through the original code unchanged. Runs on the same single tokenizer
thread as before, so the cache needs no lock. Kill switch: SEGTOK=0 in the brain's environment.

Install (brain restart required, owner window + gates; NOT done):
  1) copy segtok.py + segtok_plugin.py into a dir on the brain venv's path (e.g. /opt/1cat/tcflash-trial/ext/)
  2) register entry point group "vllm.general_plugins" name "segtok" -> segtok_plugin:register
     (a tiny dist-info in site-packages, or pip install -e of a 10-line pyproject), then restart the brain.
  3) log line "segtok: patched" appears once in the API-server log.
Rollback: SEGTOK=0 (restart) or remove the dist-info (restart).
"""
import logging
import os

log = logging.getLogger("segtok")
_MIN_CHARS = 20000
_KW_OK = {"add_special_tokens", "truncation", "max_length"}


def register():
    if os.environ.get("SEGTOK", "1") == "0":
        return
    try:
        from vllm.utils import async_utils
        from segtok import SegmentCachedEncoder
    except Exception as e:  # never break the brain over an optimisation
        log.warning("segtok: not patched (%s)", e)
        return
    cls = async_utils.AsyncMicrobatchTokenizer
    if getattr(cls, "_segtok_patched", False):
        return
    orig = cls.encode

    async def encode(self, prompt, **kwargs):
        if (isinstance(prompt, str) and len(prompt) >= _MIN_CHARS and kwargs.get("add_special_tokens") is False
                and set(kwargs) <= _KW_OK):
            enc = self.__dict__.get("_segtok")
            if enc is None:
                enc = self.__dict__["_segtok"] = SegmentCachedEncoder(self.tokenizer)
            try:
                ids = await self._loop.run_in_executor(self._executor, enc.encode, prompt)
                ml = kwargs.get("max_length")
                if not (kwargs.get("truncation") and ml is not None and len(ids) > ml):
                    return ids
            except Exception as e:
                log.warning("segtok: fallback to full encode (%s)", e)
        return await orig(self, prompt, **kwargs)

    cls.encode = encode
    cls._segtok_patched = True
    log.warning("segtok: patched AsyncMicrobatchTokenizer.encode (segment cache on <|im_start|>)")
