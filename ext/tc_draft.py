"""ThinkingCap item 2 (2026-10-26): measure graph-launch host cost vs GPU time per decode round, unprofiled.

Finding (prof-mp0 rank0 trace): the ~2.5 ms "PLE-layer wait" is NOT PLE. The 1u target graph has 1480 nodes, and its
cudaGraphLaunch takes 2.69 ms of host time (1.8 us/node; draft graphs are 60-70 nodes at 130-200 us). The GPU runs ahead
of the submission and goes idle until the launch call returns (gap end = launch return + ~90 us). It always lands at the
same graph node (a GDN layer's down_replicated_reduce -> hc_up_batch). That is why the spin-wait and FP8-lane PLE ports
changed nothing. The profiler (CUPTI) inflates launch cost, so the unprofiled size of this stall is unknown -> measure it.

Use: WEXT=tc_draft.TCDraftExt TC_DRAFT=1 (wraps tc_ext.TCExt, so ARX etc. still apply). TC_DRAFT_EVERY=N (default 2000)
replays between log lines. Per graph size it logs host replay() time p50/p90 and how many replays took > 1 ms.
Host-side timing only: no GPU sync, no graph change, no numerics change. Off (pure passthrough) unless TC_DRAFT=1.
Log line: "TC-DRAFT pid N: replay host us by nodes-bucket {...}". RPC: tc_draft_stats()."""
import os
import time

from tc_ext import TCExt

_ON = os.environ.get("TC_DRAFT", "0") == "1"
_EVERY = int(os.environ.get("TC_DRAFT_EVERY", "2000"))
_stats = {}  # key: id(graph) -> list of host us
_n = [0]


def _say(msg):
    print(f"TC-DRAFT pid {os.getpid()}: {msg}", flush=True)


def _report():
    out = {}
    for k, v in _stats.items():
        s = sorted(v)
        if s:
            out[k] = {"n": len(s), "p50": round(s[len(s) // 2]), "p90": round(s[int(len(s) * 0.9)]),
                      "gt1ms": sum(1 for x in s if x > 1000)}
    return out


def install():
    import torch

    cls = torch.cuda.CUDAGraph
    if getattr(cls, "_tc_draft_wrapped", False):
        return False
    orig = cls.replay

    def replay(self):
        t0 = time.perf_counter()
        orig(self)
        dt = (time.perf_counter() - t0) * 1e6
        lst = _stats.setdefault(id(self), [])
        if len(lst) < 4096:
            lst.append(dt)
        _n[0] += 1
        if _EVERY and _n[0] % _EVERY == 0:
            _say(f"replay host us per graph {_report()}")

    cls.replay = replay
    cls._tc_draft_wrapped = True
    return True


if _ON:
    try:
        _say(f"replay timer installed={install()}")
    except Exception as exc:  # noqa: BLE001 diagnostic only, never break the worker
        _say(f"install FAILED {exc!r}")


class TCDraftExt(TCExt):
    def tc_draft_stats(self):
        return {"enabled": _ON, "replays": _n[0], "graphs": _report()}
