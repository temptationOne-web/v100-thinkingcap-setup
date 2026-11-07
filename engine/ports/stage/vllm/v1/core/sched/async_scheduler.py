# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class AsyncScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Template for speculative decoding placeholders. Each request must get
        # its own list instance because scheduler outputs and worker feedback
        # mutate spec-token rows in later phases.
        self._spec_token_placeholders: tuple[int, ...] = (-1,) * self.num_spec_tokens
        # soundbarrier P2 (local patch, env-gated, default off): adaptive MTP
        # verification depth. With several running requests, verify only
        # VLLM_SB_ADAPTIVE_MTP_DRAFTS drafts per request (k=3 shapes) while a
        # lone request keeps the full k drafts.
        import os as _sb_os  # VLLM_SB_* read via os.environ (NOT registered in envs.py: keeps compile hash)

        _sb_d = int((_sb_os.environ.get("VLLM_SB_ADAPTIVE_MTP_DRAFTS", "0") or 0))
        self._sb_adaptive_min_reqs = max(2, int((_sb_os.environ.get("VLLM_SB_ADAPTIVE_MTP_MIN_REQS", "2") or 2)))
        self._sb_adaptive_placeholders: tuple[int, ...] | None = None
        self._sb_adaptive_max_ctx = int((_sb_os.environ.get("VLLM_SB_ADAPTIVE_MTP_MAX_CTX", "0") or 0))
        if 0 < _sb_d < self.num_spec_tokens:
            self._sb_adaptive_placeholders = (-1,) * _sb_d
            logger.info(
                "soundbarrier adaptive MTP depth ON: %d drafts with 1 running "
                "request, %d drafts with >= %d running requests.",
                self.num_spec_tokens,
                _sb_d,
                self._sb_adaptive_min_reqs,
            )
        self.pp_size = self.parallel_config.pipeline_parallel_size
        speculative_config = self.vllm_config.speculative_config
        draft_model_config = (
            speculative_config.draft_model_config
            if speculative_config is not None
            else None
        )
        self._drafter_max_model_len = (
            draft_model_config.max_model_len
            if draft_model_config is not None
            and draft_model_config.max_model_len is not None
            else self.max_model_len
        )

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_after_schedule(scheduler_output)
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        spec_placeholders = self._spec_token_placeholders
        if (
            self._sb_adaptive_placeholders is not None
            and len(self.running) >= self._sb_adaptive_min_reqs
        ):
            spec_placeholders = self._sb_adaptive_placeholders
        elif (
            self._sb_adaptive_placeholders is not None
            and self._sb_adaptive_max_ctx > 0
            and len(self.running) == 1
            and self.running[0].num_tokens > self._sb_adaptive_max_ctx
        ):
            # A lone long-context request: the extra verifier row costs more
            # attention than its expected accepted token is worth.
            spec_placeholders = self._sb_adaptive_placeholders
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            # The request will generate a new token plus num_spec_tokens
            # in this scheduling step.
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            request.num_output_placeholders += 1 + cur_num_spec_tokens
            # Add placeholders for the new draft/spec tokens.
            # We will update the actual spec token ids in the worker process.
            if (
                request.num_computed_tokens + self.num_spec_tokens
                <= self._drafter_max_model_len
            ):
                request.spec_token_ids = list(spec_placeholders)
            else:
                request.spec_token_ids = []

            if self.use_v2_model_runner:
                request.next_decode_eligible_step = self.current_step + self.pp_size

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        if request.async_tokens_to_discard > 0:
            # The request was force-preempted in reset_prefix_cache; drop one
            # stale in-flight async output frame per call until the counter
            # is drained.
            request.async_tokens_to_discard -= 1
            return [], False

        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )

        # Update the number of output placeholders.
        request.num_output_placeholders -= len(new_token_ids)
        assert request.num_output_placeholders >= 0

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped
