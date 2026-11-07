# ThinkingCap-Qwen3.8-Flash-Next NVFP4 on 4× V100 32GB (PCIe, no NVLink)

**What this is:** a working recipe for running a modern 2026 MoE model, ThinkingCap-Qwen3.8-Flash-Next (NVFP4), fast on
**old Tesla V100 GPUs** (Volta, SM70) with **no NVLink**, using the 1Cat-vLLM fork. It is my home AI server's real
production setup (it runs my personal assistant every day), with every setting, patch and benchmark I used.

**Who it is for:** anyone with 2–4 V100 32GB cards (or similar Volta boxes) who wants a smart, fast local model with
vision, tool calling and a 262k context, and wants to skip months of trial and error. You can hand this README and
the repo to your AI coding assistant and ask it to reproduce the setup.

**What you get on 4× V100 (measured):** ~160–170 tok/s for one user, ~340 tok/s total for four users,
LiveCodeBench easy+medium 55/61, 262k context, tool calls and JSON output working.

**What is NOT here:** the model weights (download from Hugging Face, link below), and nothing private.

### Quick start (most of the speed, least work)
1. Hardware/OS: 4× V100 32GB, NVIDIA driver 580.x, CUDA 12.8, add `iommu=pt` to the kernel command line.
2. Download the model: https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-Flash-Next-NVFP4 (~100 GB).
3. Build 1Cat-vLLM main (commit `e098ee11b` or newer) for SM70: see `engine/build.sh` (~1 hour on my 36-thread Xeon).
4. Apply at least the CT-NVFP4 patch (`engine/ports/stage/.../compressed_tensors_moe_w4a4_nvfp4.py`) and set `VLLM_SM70_CT_NVFP4_TM=1`.
5. Start with the command and environment in section 4. The first start compiles kernels and takes a while; later starts took me ~8 minutes.
6. Then add the other patches and plug-ins from this repo one at a time, checking speed and output after each.

### How long it took me
| Stage | Time |
|---|---|
| Learning V100 + 1Cat-vLLM: kernels, quant formats, MTP, PCIe all-reduce, across several earlier models (Qwen3.8-27B, Swift-1.5 Flash-Next 122B, GLM-5.3-Flash, Ornith-1.5-122B) | **~4.5 months** (June → October 2026). Most patches here were born in that period. |
| ThinkingCap from first download to a fast, usable server | **~1 day** (25 Oct evening → 26 Oct). The big fix was routing its NVFP4 experts to the tuned TurboMind path (80 → 114 tok/s for 1 user). |
| Making it production-safe (JSON/grammar 500 errors, tool-call parser, scheduler, reproducibility) | **~1 day** (27 Oct) |
| Extra speed work (no-sync sampler, long-chat tokenizer reuse, engine upgrade to 1Cat e098ee11b, FLy) | **~10 days**, on and off (27 Oct → 6 Nov) |
| **Total for this exact setup** | **~2 weeks**, on top of the earlier months of V100 groundwork |

With this repo, someone with the same hardware should get the quick-start version running in **an afternoon**
(mostly the engine build and the 100 GB download), and the full setup in **a day or two**.

---

## Summary

- **Hardware**: 4 × Tesla V100-SXM2 **32GB**, all on **PCIe 3.0 x8**, **no NVLink** (PHB between GPUs). Xeon E5-2699 v3, 128 GB RAM, MSI X99A. GPUs capped at 150 W. Kernel flag `iommu=pt` (+11% P2P).
- **Model**: `bottlecapai/ThinkingCap-Qwen3.8-Flash-Next-NVFP4` (Qwen3.8 Flash-Next MoE, ~6B active, vision, own MTP head).
- **Engine**: 1Cat-vLLM main (`e098ee11b`, 2026-11-05) + my 12 patches + FLy. TP4, 262,144 context.
- **Speed** (re-measured 2026-11-07 on the live server): ~160–170 tok/s for 1 user, ~250 total for 2, ~340 total for 4 (up to ~370 right after a fresh boot). Reads a 54k-token prompt cold in ~17–19 s (real text).
- **Coding**: LiveCodeBench easy+medium **55/61**.
- **Most important patch**: `VLLM_SM70_CT_NVFP4_TM=1` (compressed-tensors NVFP4 experts use 1Cat's tuned SM70 TurboMind route instead of Marlin): 1 user 80 → 114 tok/s.
- **Compared**: Ripple-65B-Preview (dense, NVFP4A16, same box, TP4) did 37 tok/s and 47/61, so slower and weaker.

---

## 1. Hardware

| Part | What I have |
|---|---|
| GPUs | 4 × Tesla V100-SXM2-32GB (128 GB total) |
| GPU links | **PCIe 3.0 x8 each, NO NVLink**. `nvidia-smi topo -m` shows PHB between every pair |
| Power | GPUs capped at **150 W** each (I had power-supply trips at 190 W) |
| CPU / board / RAM | Xeon E5-2699 v3 (36 threads), MSI X99A, 128 GB DDR4 |
| Driver / CUDA | NVIDIA 580.178.04, CUDA 12.8 |
| Kernel flag | `iommu=pt` on the kernel command line: **+11% P2P / all-reduce speed** for free on this board |
| Disk | NVMe for the engine + model (the PLE lookup table is ~51 GB) |

## 2. Model

- **`bottlecapai/ThinkingCap-Qwen3.8-Flash-Next-NVFP4`** (Hugging Face).
- Qwen3.8 Flash-Next architecture (`qwen4_exp`): Mixture-of-Experts, 512 experts, 10 active, 48 layers, hidden 2560, ~6B active per token, vision, MTP head, plus a large n-gram "PLE" embedding table.
- NVFP4 in **compressed-tensors** format.

## 3. Engine

- **1Cat-vLLM main**, commit **`e098ee11b`** (2026-11-05), built for SM70 only. See `engine/build.sh` (key env: `TORCH_CUDA_ARCH_LIST=7.0`, `CMAKE_CUDA_ARCHITECTURES=70`, `FLASH_ATTN_V100_CUDA_ARCH_LIST=7.0`, CUDA 12.8).
- Python 3.12 venv. Exact package versions are in `engine/freeze.txt` (torch 2.10.0+cu128). Install with `uv pip install --no-deps -r freeze.txt`, then the wheel.
- Then my patches (§5): `SKIP_WHEEL=1 bash engine/ports/apply-ports.sh`. It refuses to patch if the installed file differs from what it expects. Read it first and edit the paths in it.

**Note:** patches go into `.venv/lib/python3.12/site-packages/vllm/...`, the installed copy, not the `src/` tree. I lost two days once by patching `src/`, which the server never imports.

## 4. Launch command (exactly what runs now)

```bash
vllm serve /path/to/ThinkingCap-Qwen3.8-Flash-Next-NVFP4 \
  --served-model-name thinkingcap \
  --trust-remote-code --dtype half --kv-cache-dtype auto \
  --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 4 --gpu-memory-utilization 0.905 \
  --max-model-len 262144 --max-num-seqs 4 --max-num-batched-tokens 4096 \
  --enable-prefix-caching --mamba-cache-mode align --partial-prefix-cache \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4,"draft_sample_method":"probabilistic","use_local_argmax_reduction":false,"disable_eagle_block_drop":true,"attention_backend":"FLASH_ATTN_V100"}' \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --scheduling-policy priority --mixed-prefill-step-latency-ms 0 \
  --async-scheduling
# mine on top (optional, need ext/ on PYTHONPATH):
#  --worker-extension-cls tc_prod_ext.TCProdExt
#  --scheduler-cls tc_solo_sched.TCSoloStepScheduler
#  --tool-parser-plugin ext/tc_toolfix.py --tool-call-parser qwen3_coder_tcfix
```

### Environment (the ones that matter)

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_IB_DISABLE=1
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_V2_CUDAGRAPH_MEM_MIB=460

# --- biggest single win: CT-format NVFP4 experts -> 1Cat's tuned SM70 TurboMind MoE route (my port #8) ---
export VLLM_SM70_CT_NVFP4_TM=1            # 1 user: 80 -> 114 tok/s. Without it the experts fall back to Marlin.
export VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES=0   # autotune crashed at warm-up for me
export VLLM_SM70_NVFP4_QWEN38_MOE_QPN_DYNAMIC_DECODE=1   # batched expert route for 2..16 verify tokens

# --- Qwen3.8 Flash-Next fast paths (1Cat built-ins + my packed-only port) ---
export VLLM_SM70_QWEN38_HYBRID_PLE=1      # PLE table split GPU/CPU. Disk-PLE was 25-42% slower on real text for me
export VLLM_PLE_CPU_OFFLOAD=1 VLLM_PLE_DISK_OFFLOAD=1
export VLLM_SM70_QWEN38_PACKED_ONLY=1 VLLM_SM70_QWEN38_FP16_GEMV=1
export VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16=1 VLLM_SM70_QWEN38_FUSED_HC_FP16=1
export VLLM_SM70_HC_FORCE_REPLICATED=1
export VLLM_FLASH_V100_DECODE_PARTITION_SIZE=256

# --- PCIe without NVLink ---
export VLLM_SM70_PCIE_P2P_AS_FULL=1       # my port: treat PCIe P2P as usable for custom all-reduce (needs iommu=pt)

# --- MTP speculative decoding ---
export VLLM_SM70_MTP_ROUTER_BATCH=1 VLLM_SM70_MTP_SHARED_BATCH=1 VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT=0
export VLLM_SB_ADAPTIVE_MTP_DRAFTS=3 VLLM_SB_ADAPTIVE_MTP_MAX_CTX=24000   # my adaptive-MTP port: fewer drafts on long chats

# --- FLy (my port of upstream vLLM PR #53987, lossy verification, only for temperature > 0) ---
export TC_FLY_W=1 TC_FLY_THR=0.15 TC_FLY_TOPK=3   # +9% 1 user, +17% 4 users on normal chat sampling; same coding score
```

My full launcher is in `launcher/run-tc-prod.sh` → `run-tc-best.sh` → `run-tc-new.sh`, plus the systemd drop-ins in `launcher/systemd-dropins/` (read them in number order; later files override earlier ones). Paths are mine (`/opt/1cat/...`); change them.

## 5. What is in this pack

| Folder | What | Needed? |
|---|---|---|
| `engine/ports/stage/` | 12 patched vLLM files: CT-NVFP4 → TurboMind route, packed-only + FP16 GEMV/HC for Qwen3.8, PCIe-P2P-as-full all-reduce, adaptive MTP (arg_utils, async_scheduler, cudagraph_utils, compilation, speculator), 1Cat #567 offload alloc fix, ReVolta #74 (num_warps=1 in the rejection sampler) | **Yes**, for my speed |
| `engine/fly/rejection_sampler_utils.py` | The FLy version of the rejection sampler (replaces the staged one) | Optional (+9–17%) |
| `engine/RECIPE.txt`, `engine/build.sh`, `engine/freeze.txt` | How I build and install | Yes |
| `ext/` | My runtime plug-ins (no engine rebuild): `tc_glue` (fused layer-boundary step, +3–5%), `tc_nosync` (sampler no CPU/GPU sync, +25% at temp 1), `segtok` (re-uses tokenization of long chats), `tc_toolfix` (qwen3_coder tool-call parser fixes), `tc_sodraft_fix` (JSON/grammar requests with MTP drafts, fixes HTTP 500s), `tc_kvfix`, `tc_solo_sched` (priority lane scheduler) | Optional, each one is switchable |
| `sm70-extras/` | Older speed hooks my extension chain still imports: `sbcar` (all-reduce), `sbfuse` (fused MTP MoE, **build `sb_fuse_kernels.cu` yourself for sm_70**, I removed the .so), `sbp5` (OFF in my config), Ornith scheduler/kv files | Only if you use `ext/tc_prod_ext.py` |
| `launcher/` | My launch scripts + systemd drop-ins | Reference |

**Easiest start:** stock 1Cat main + the launch command in §4 + `VLLM_SM70_CT_NVFP4_TM=1` (needs the `compressed_tensors_moe_w4a4_nvfp4.py` port) + `iommu=pt`. That gets most of the speed. Add `ext/` pieces one at a time after that.

## 6. Results (all on this machine, GPUs at 150 W)

### Speed: writing (decode)

| Test | 1 user | 2 users (total) | 4 users (total) |
|---|---|---|---|
| Chat sampling (temp 1, top_k 20, top_p 0.95), FLy ON, live server up 4 h (2026-11-07) | **158 median / 170 mean** | 252 | **336** |
| Same, right after a fresh boot (2026-11-06) | 162 | 242 | 371 |
| Fresh boot, FLy OFF | 149 | 232 | 317 |
| Greedy (temp 0) | ~160 | — | — |
| Real mixed text prompts, hardest case (low-repeat text) | 115–124 | 184 | 244 |

MTP with 4 draft tokens, ~2.1 accepted per draft with FLy.

### Speed: reading (prefill), cold (nothing cached)

| Prompt | Time to first token |
|---|---|
| 13.5k tokens | 4.0 s fresh boot / 4.7 s live (≈2,900 tok/s) |
| 54k tokens | 16.6 s fresh boot / 19.4 s live (≈2,750 tok/s) |
| 126k tokens | 40.7 s fresh boot |
| 54k tokens of random dictionary words (worst case for the PLE table) | 23.1 s live |
| Follow-up in a 44k-token chat (prefix cached) | ~0.4 s |

Context 262,144 tokens. GPU KV pool 339,200 tokens shared by all requests.

### Quality

| Test | Result |
|---|---|
| LiveCodeBench, easy + medium, 61 problems (32k max tokens) | **55/61** (easy 28/28, medium 27/33), same with FLy on or off |
| Same 61, the 6–8 misses re-tried with a longer "deep solve" loop (generate + run tests + retry, up to 20 min) | 8/8 of the misses solved |
| LCB sample, ThinkingCap vs Ornith-1.5-122B (my previous brain) | 23/32 vs 16/32 |
| My hard reasoning set | 21/21, ~1.6 s per answer |
| Tool calling (my 15 agent tool tests) | 15/15 |
| JSON-schema soak (15 min, mixed traffic) | 0 errors |

### Other models I tried on the same 4× V100 (for comparison)

| Model | Engine | 1-user speed | LCB easy+med | Verdict |
|---|---|---|---|---|
| **ThinkingCap-Qwen3.8-Flash-Next NVFP4** | 1Cat-vLLM TP4 | **~150–162** | **55/61** | **My choice** |
| Ornith-1.5-122B (uncensored) | 1Cat-vLLM TP4 | ~170 | worse (16/32 vs 23/32 sample) | faster but weaker at code |
| Ripple-65B-Preview (dense 65B), NVFP4A16 | 1Cat-vLLM TP4 (I ported the ricursive arch) | 37 (4 users: ~120 total) | 47/61 | slower and weaker |
| Ripple-65B-Preview Q4_K_M | llama.cpp, `-sm tensor -fa on` | 23–24 | — | slow |
| Ripple-65B + DFlash2 drafter | llama.cpp | 21–23 code, 9–10 chat | — | drafter hurts chat |

## 7. Lessons (things that cost me time)

1. **Patch site-packages, not `src/`.** Check with `python -c "import vllm; print(vllm.__file__)"` run from a different folder.
2. **`iommu=pt`** on a no-NVLink board was a free +11%.
3. **Compressed-tensors NVFP4 MoE goes to Marlin by default on SM70.** The tuned 1Cat route only takes ModelOpt checkpoints unless you bridge it (my `VLLM_SM70_CT_NVFP4_TM` port).
4. **Speed A/Bs need fresh boots, same engine.** A server that has been up for hours measured 129 tok/s vs 149 fresh, which looked like a fake +16%.
5. **Benchmark with real, varied text.** Repeating similar prompts keeps the PLE rows cached and hides a 25–42% slowdown from disk-PLE.
6. **RAM chat cache (KV offload)** works on engine e098ee11b only with my old kv-shift patch off (`ORN_MTP_KVFIX=0`). I keep it off because hybrid-PLE needs the RAM.
7. **Gate every change** with tool-call tests, a JSON-schema soak and a reproducibility check, not only `/health`. One speed patch passed every speed test and still broke JSON requests in production.
8. **DFlash/DFlash2 drafters lost to the model's own MTP head** every time on V100 for me.

Questions: ask JJ in the group.
