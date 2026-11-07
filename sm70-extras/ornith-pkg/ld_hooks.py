import torch
"""layer_diff.py worker-side hooks (importable so the engine can pickle them by reference). Measurement only."""
def install_hooks(model):
    import torch as _t
    st = {"on": False, "rows": []}
    model._ld_cap = st

    def mk(name):
        def hook(mod, inp, out):
            if not st["on"]:
                return
            o = out[0] if isinstance(out, (tuple, list)) else out
            if not isinstance(o, _t.Tensor) or o.dim() < 2:
                return
            if o.shape[0] > 4:  # prefill call: skip
                return
            st["rows"].append((name, int(o.shape[0]), o[0].detach().float().cpu().clone()))
        return hook

    n = 0
    deep = tuple("layers.%d." % i for i in (11, 12, 27))
    for name, mod in model.named_modules():
        if "visual" in name:
            continue
        top = name.endswith(tuple(f"layers.{i}" for i in range(48))) or name.endswith(("model.norm", "logits_processor"))
        if top or any(d in name for d in deep):
            mod.register_forward_hook(mk(name))
            n += 1
    return n


def set_on(model, on):
    st = model._ld_cap
    st["on"] = on
    if on:
        st["rows"] = []
    return True


def fetch(model):
    import torch.distributed as dist
    st = model._ld_cap
    r = dist.get_rank() if dist.is_initialized() else 0
    if r != 0:
        return None
    import os as _os
    path = "/opt/1cat/ornith122-trial/barrier/logs/ld_rows_%d.pt" % _os.getpid()
    torch.save(st["rows"], path)
    return path


