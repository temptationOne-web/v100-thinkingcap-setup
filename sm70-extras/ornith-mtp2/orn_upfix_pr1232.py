"""Upstream backport (2026-10-25, owner "do the small fix"): PR #1232 (open upstream), "Finish reused-block stores before
MRv2 zeroes them".

With the native OffloadingConnector (our RAM chat cache) each step's GPU->CPU stores are deferred to the start of the
next step; stores whose source block has been handed out again are listed in jobs_to_flush and must finish before the
block is reused. The V2 model runner did that submit-and-wait in ActiveKVConnector.pre_forward -> handle_preemptions,
which runs AFTER execute_model has already zeroed the step's new blocks (new_block_ids_to_zero) and run the Mamba
align preprocess_state. The store stream waits on the compute stream at submit time, so such a store copied ZEROS into
the CPU tier, which later prefix hits would load (silent: zero KV / GDN state). MRv1 is not affected.

Fix (as upstream): call handle_preemptions right after the step's request/block-table updates, before the zeroing;
pre_forward skips the repeat for the same metadata. Installed from the worker extension (ornith_ext_cp1) before the
model runner is built; patches classes, edits no vLLM file. Nothing here is traced by torch.compile.

Placement: upstream calls it after block_tables.apply_staged_writes(); we call it at the end of update_requests(), which
is immediately before apply_staged_writes() in execute_model (non-dummy path only). apply_staged_writes only uploads
block-table rows (no KV data), so the store/zero ordering is identical.
"""
from __future__ import annotations

import os

ENABLED = os.environ.get("ORN_UPFIX_PR1232", "1") == "1"
STATS = {"early": 0, "dedup_skips": 0}
_INSTALLED = False


def install() -> bool:
    global _INSTALLED
    if _INSTALLED or not ENABLED:
        return _INSTALLED
    from vllm.v1.worker.gpu import kv_connector as kc
    from vllm.v1.worker.gpu import model_runner as mr

    def base_handle_preemptions(self, scheduler_output) -> None:
        pass

    def handle_preemptions(self, scheduler_output) -> None:
        if self._disabled:
            return
        md = scheduler_output.kv_connector_metadata
        assert md is not None
        self.kv_connector.handle_preemptions(md)
        self._orn_handled_md = md
        STATS["early"] += 1

    def pre_forward(self, scheduler_output) -> None:
        if self._disabled:
            return
        md = scheduler_output.kv_connector_metadata
        assert md is not None
        if getattr(self, "_orn_handled_md", None) is not md:
            self.kv_connector.handle_preemptions(md)
        else:
            STATS["dedup_skips"] += 1
        self._orn_handled_md = None
        self.kv_connector.bind_connector_metadata(md)
        if kc.is_forward_context_available():
            self.kv_connector.start_load_kv(kc.get_forward_context())
        else:
            with kc.set_forward_context(None, self.vllm_config):
                self.kv_connector.start_load_kv(kc.get_forward_context())

    kc.KVConnector.handle_preemptions = base_handle_preemptions
    kc.ActiveKVConnector.handle_preemptions = handle_preemptions
    kc.ActiveKVConnector.pre_forward = pre_forward

    orig_update_requests = mr.GPUModelRunner.update_requests

    def update_requests(self, scheduler_output) -> None:
        orig_update_requests(self, scheduler_output)
        self.kv_connector.handle_preemptions(scheduler_output)

    update_requests._orn_upfix_pr1232 = True
    mr.GPUModelRunner.update_requests = update_requests
    _INSTALLED = True
    return True


def status() -> dict:
    return {"installed": _INSTALLED, **STATS}
