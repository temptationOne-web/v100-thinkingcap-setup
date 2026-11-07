"""qwen3_coder_tcfix: chunking-proof streaming for the qwen3_coder tool parser (2026-10-27).

BUG (engine /opt/1cat/vllm-main-20261025, vllm/tool_parsers/qwen3coder_tool_parser.py):
  extract_tool_calls_streaming() is a state machine that advances ONE step per call and returns early
  (header -> return; "{" -> return; params -> return; "}" -> return). It assumes every step gets its own delta.
  Under load the API server merges several engine steps into one delta, so the text that should drive the
  remaining steps arrives together and the steps after the first are never run:
    no-arg tool:  delta A = "<tool_call>\\n<function=get_time>\\n"  -> header, arguments ""
                  delta B = "</function>\\n</tool_call>"            -> returns "{" (the json_started branch,
                            line ~776) and stops; the "}" step needs another delta with text, which never comes.
                  delta C = "" + EOS (finish_reason)               -> empty-delta branch returns content=""
  Result: streamed arguments "{" while non-streamed arguments are "{}".
  The end-of-stream catch-up in chat_completion/serving.py (~line 836) does not repair it: it only runs when the
  FINAL delta itself carries tool_call arguments (_should_check_for_unstreamed_tool_arg_tokens), so a final
  empty/EOS delta skips it. When it does run, it only looks at the LAST tool index, measures the "latest delta"
  from tool_calls[0] (possibly another tool) and REPLACES the whole delta message (dropping other tools' fragments
  and content in that delta).

FIX:
  Qwen3CoderTCFixToolParser (registered as "qwen3_coder_tcfix") recomputes, on every call, the complete target
  state from current_text with the SAME regexes/helpers the non-streamed extract_tool_calls() uses, and emits only
  the difference to what was already streamed. Arguments of a finished function are exactly
  _parse_xml_function_call(...).function.arguments; while a function is still open only "{" plus fully delimited
  parameters (those followed by the next real parameter tag) are streamed, which is always a prefix of the final
  JSON. So chunking (1 token, merged bursts, whole output at once, tool end + finish in the same delta, several
  tools in one delta) cannot change the result.
  Serving safety net: _should_check_for_unstreamed_tool_arg_tokens is wrapped. For a tcfix parser at the finish
  chunk it (1) asks the parser to finalize (an unclosed call at max_tokens is parsed like the non-streamed path),
  (2) for EVERY tool index appends expected[len(streamed):] when the streamed text is a proper prefix of
  prev_tool_call_arr[i]["arguments"], merging into the existing delta, and returns False so the original
  replace-the-delta catch-up never runs. Other parsers keep the original behaviour.

Use: --tool-parser-plugin /opt/1cat/tcflash-trial/ext/tc_toolfix.py --tool-call-parser qwen3_coder_tcfix
(run-tc-prod.sh adds this when TC_TOOLFIX=1). Log line: "TC-TOOLFIX ... serving safety net installed=True".
Offline test: /opt/1cat/tcflash-trial/ext/tests/test_tc_toolfix_cpu.py
"""
import json
import os
import sys
from collections.abc import Sequence

from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.logger import init_logger
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser
from vllm.tool_parsers.utils import find_tool_properties

logger = init_logger(__name__)


def _say(msg):
    print(f"TC-TOOLFIX pid {os.getpid()}: {msg}", flush=True)


@ToolParserManager.register_module("qwen3_coder_tcfix", force=True)
class Qwen3CoderTCFixToolParser(Qwen3CoderToolParser):
    """qwen3_coder with diff-based (chunking independent) streaming."""

    _tcfix = True

    def _reset_streaming_state(self):
        super()._reset_streaming_state()
        self._tc_mode = None          # None | "tag" (<tool_call> wrapped) | "bare" (<function= only)
        self._tc_content_sent = 0     # chars of current_text already emitted as content
        self._tc_units = []           # per discovered call unit: tool index (int) | None (invalid) ; ordered
        self._tc_unit_done = []       # per unit: True once its final arguments were emitted
        self._tc_last_text = ""
        self._tc_finalized = False

    # ------------------------------------------------------------------ helpers
    def _tc_partial_args(self, func_content: str) -> str:
        """'{' + fully delimited params of an OPEN function; always a prefix of the final JSON."""
        name_end = func_content.find(">")
        function_name = func_content[:name_end]
        param_config = find_tool_properties(self.tools, function_name)
        parameters = func_content[name_end + 1:]
        starts = self._parameter_start_positions(parameters, param_config)
        frags, seen = [], set()
        for k, position in enumerate(starts):
            if k + 1 >= len(starts):
                break  # the last parameter is only known when the function closes
            name_start = position + len(self.parameter_prefix)
            pname_end = parameters.find(">", name_start)
            if pname_end == -1:
                break
            pname = parameters[name_start:pname_end]
            if pname in seen:
                break  # duplicate key: the final dict collapses it -> wait for the close
            seen.add(pname)
            value_start = pname_end + 1
            next_start = starts[k + 1]
            value_end = self._find_structural_parameter_end(
                parameters, value_start=value_start, next_parameter_start=next_start,
                container_end=len(parameters), allow_text_end=True)
            if value_end is None:
                value_end = next_start
            value = parameters[value_start:value_end]
            if value.startswith("\n"):
                value = value[1:]
            if value.endswith("\n"):
                value = value[:-1]
            conv = self._convert_param_value(value, pname, param_config, function_name)
            frags.append(json.dumps(pname, ensure_ascii=False) + ": " + json.dumps(conv, ensure_ascii=False))
        return "{" + ", ".join(frags)

    def _tc_iter_units(self, text: str, at_end: bool):
        """Yield ("xml", func_content, final) / ("json", block_content, True) / ("wait", ...) in output order,
        mirroring extract_tool_calls(): blocks = tool_call_regex, functions = tool_call_function_regex."""
        if self._tc_mode is None:
            if self.tool_call_start_token in text:
                self._tc_mode = "tag"
            elif self.tool_call_prefix in text:
                self._tc_mode = "bare"
            else:
                return
        if self._tc_mode == "tag":
            for m in self.tool_call_regex.finditer(text):
                closed = m.group(1) is not None
                body = m.group(1) if closed else m.group(2)
                fms = list(self.tool_call_function_regex.finditer(body))
                if not fms:
                    if self.tool_call_prefix in body:  # cannot happen (regex finds it); be safe
                        yield ("wait", None, False)
                        return
                    if closed:
                        yield ("json", body, True)
                        continue
                    if at_end:
                        return  # non-streamed path ignores an unclosed JSON body
                    yield ("wait", None, False)
                    return
                for fm in fms:
                    if fm.group(1) is not None:
                        yield ("xml", fm.group(1), True)
                    else:
                        yield ("xml", fm.group(2), closed or at_end)
                if not closed:
                    return  # nothing can follow an open block
        else:  # bare <function=...> blocks, no <tool_call> wrapper
            for fm in self.tool_call_function_regex.finditer(text):
                if fm.group(1) is not None:
                    yield ("xml", fm.group(1), True)
                else:
                    yield ("xml", fm.group(2), at_end)

    def _tc_first_tool_pos(self, text: str) -> int:
        cands = [p for p in (text.find(self.tool_call_start_token), text.find(self.tool_call_prefix)) if p != -1]
        if not cands:
            return -1
        a = text.find(self.tool_call_start_token)
        # same rule as extract_tool_calls(): <tool_call> position if present, else <function=
        return a if a != -1 else min(cands)

    def _tc_process(self, text: str, at_end: bool) -> DeltaMessage | None:
        content = None
        tpos = self._tc_first_tool_pos(text)
        if tpos == -1:
            end = len(text) if at_end else len(text) - len(self._trailing_tool_marker_prefix(text))
        else:
            end = tpos
        if end > self._tc_content_sent:
            content = text[self._tc_content_sent:end]
            self._tc_content_sent = end

        out: dict[int, DeltaToolCall] = {}
        order: list[int] = []

        def add(idx, args, header=None):
            if idx not in out:
                out[idx] = DeltaToolCall(index=idx, function=DeltaFunctionCall(arguments=""))
                order.append(idx)
            tc = out[idx]
            if header is not None:
                tc.id, tc.type = header[0], "function"
                tc.function.name = header[1]
            tc.function.arguments = (tc.function.arguments or "") + args

        for u, (kind, body, final) in enumerate(self._tc_iter_units(text, at_end)):
            if kind == "wait":
                break
            if u < len(self._tc_unit_done) and self._tc_unit_done[u]:
                continue
            if u >= len(self._tc_units):
                # new unit: resolve its header
                if kind == "json":
                    parsed = self._parse_json_function_call(body)
                    if parsed is None:
                        self._tc_units.append(None)
                        self._tc_unit_done.append(True)
                        continue
                    name, target = parsed.function.name, parsed.function.arguments
                else:
                    name_end = body.find(">")
                    if name_end == -1:
                        if final:  # non-streamed parse returns None for this one
                            self._tc_units.append(None)
                            self._tc_unit_done.append(True)
                            continue
                        break  # header still incomplete: wait (keeps later units in order)
                    name, target = body[:name_end], None
                idx = len(self.prev_tool_call_arr)
                self.prev_tool_call_arr.append({"name": name, "arguments": "{}"})
                self.streamed_args_for_tool.append("")
                self._tc_units.append(idx)
                self._tc_unit_done.append(False)
                tool_id = self._generate_tool_call_id()
                header = (tool_id, name)
                self.current_tool_index = idx
            else:
                idx = self._tc_units[u]
                if idx is None:
                    continue
                header, target = None, None
                if kind == "json":
                    target = self._parse_json_function_call(body).function.arguments

            if target is None:
                if final:
                    parsed = self._parse_xml_function_call(body)
                    target = parsed.function.arguments if parsed is not None else "{}"
                else:
                    target = self._tc_partial_args(body)
            if final:
                self.prev_tool_call_arr[idx]["arguments"] = target
                self._tc_unit_done[u] = True
            streamed = self.streamed_args_for_tool[idx]
            if target.startswith(streamed):
                delta = target[len(streamed):]
            else:
                delta = ""
                logger.warning("qwen3_coder_tcfix: streamed args are not a prefix of the %s target "
                               "(tool %d); not re-emitting", "final" if final else "partial", idx)
            if delta:
                self.streamed_args_for_tool[idx] = streamed + delta
            if delta or header is not None:
                add(idx, delta, header)

        if content is None and not order:
            return None
        msg = DeltaMessage(content=content) if content is not None else DeltaMessage()
        if order:
            msg.tool_calls = [out[i] for i in order]
        return msg

    # ------------------------------------------------------------------ API
    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request,
    ) -> DeltaMessage | None:
        if not previous_text:
            self._reset_streaming_state()
            self.streaming_request = request
        self._tc_last_text = current_text
        if not delta_text:
            return None
        return self._tc_process(current_text, at_end=False)

    def _tcfix_finalize(self) -> DeltaMessage | None:
        """End of stream: finish unclosed calls like the non-streamed parse, flush held content."""
        if self._tc_finalized:
            return None
        self._tc_finalized = True
        return self._tc_process(self._tc_last_text, at_end=True)


# ---------------------------------------------------------------------------- serving safety net
def _merge_into(delta_message: DeltaMessage, extra: DeltaMessage | None):
    if extra is None:
        return
    if extra.content:
        delta_message.content = (delta_message.content or "") + extra.content
    if extra.tool_calls:
        tcs = list(delta_message.tool_calls or [])
        for etc in extra.tool_calls:
            cur = next((t for t in tcs if t.index == etc.index), None)
            if cur is None:
                tcs.append(etc)
                continue
            if cur.function is None:
                cur.function = DeltaFunctionCall(arguments="")
            if etc.id and not cur.id:
                cur.id, cur.type = etc.id, "function"
            if etc.function and etc.function.name and not cur.function.name:
                cur.function.name = etc.function.name
            if etc.function and etc.function.arguments:
                cur.function.arguments = (cur.function.arguments or "") + etc.function.arguments
        delta_message.tool_calls = tcs


def tcfix_finish(tool_parser, delta_message: DeltaMessage) -> None:
    """Finalize the parser and append, for every tool, whatever of the expected arguments was not streamed."""
    _merge_into(delta_message, tool_parser._tcfix_finalize())
    extra = []
    for idx, call in enumerate(tool_parser.prev_tool_call_arr):
        if idx >= len(tool_parser.streamed_args_for_tool):
            break
        expected = call.get("arguments", "{}")
        if not isinstance(expected, str):
            expected = json.dumps(expected, ensure_ascii=False)
        streamed = tool_parser.streamed_args_for_tool[idx]
        if expected != streamed and expected.startswith(streamed):
            rem = expected[len(streamed):]
            tool_parser.streamed_args_for_tool[idx] = expected
            extra.append(DeltaToolCall(index=idx, function=DeltaFunctionCall(arguments=rem)))
        elif expected != streamed:
            logger.warning("qwen3_coder_tcfix: tool %d streamed args diverge from the parsed args", idx)
    if extra:
        _merge_into(delta_message, DeltaMessage(tool_calls=extra))


def install_serving_safety_net() -> bool:
    import inspect

    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

    cur = OpenAIServingChat._should_check_for_unstreamed_tool_arg_tokens
    if getattr(cur, "_tcfix", False):
        return True
    src = inspect.getsource(OpenAIServingChat.chat_completion_stream_generator)
    if "tool_parser = parser.tool_parser" not in src or "self._should_check_for_unstreamed_tool_arg_tokens(" not in src:
        _say("serving layout changed: safety net NOT installed (parser fix still active)")
        return False
    orig = cur

    def _should_check(self, delta_message, output):
        try:
            if output.finish_reason is not None and delta_message is not None:
                tp = sys._getframe(1).f_locals.get("tool_parser")
                if tp is not None and getattr(tp, "_tcfix", False):
                    tcfix_finish(tp, delta_message)
                    return False  # never run the replace-the-delta catch-up for tcfix parsers
        except Exception:  # noqa: BLE001  the safety net must never break a stream
            logger.exception("qwen3_coder_tcfix safety net failed")
        return orig(self, delta_message, output)

    _should_check._tcfix = True
    _should_check.__wrapped__ = orig
    OpenAIServingChat._should_check_for_unstreamed_tool_arg_tokens = _should_check
    return True


if os.environ.get("TC_TOOLFIX_NO_SERVING_PATCH", "0") != "1":
    try:
        _say(f"parser qwen3_coder_tcfix registered; serving safety net installed={install_serving_safety_net()}")
    except Exception as exc:  # noqa: BLE001
        _say(f"serving safety net install FAILED {exc!r} (parser fix still active)")
