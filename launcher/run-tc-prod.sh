#!/bin/bash
# run-tc-prod.sh (2026-10-26, PREPARED for the Ornith -> ThinkingCap switch; inert until 40-tc.conf is installed).
# JARVIS PROD launcher for ThinkingCap-Qwen3.8-Flash-Next-NVFP4 on :8030, started by unit 1cat-ornith-brain via
# /etc/systemd/system/1cat-ornith-brain.service.d/40-tc.conf. It DERIVES from the speed investigator's launcher so
# last-minute changes carry over:
#   base = run-tc-best.sh (else run-tc-new.sh). At every start it writes copies into golive/gen/:
#     run-tc-new.prod.sh  = run-tc-new.sh with VLLM_CACHE_ROOT / MEMGUARD_LOG made overridable and ${TC_PROD_ARGS}
#                           added before --async-scheduling
#     run-tc-best.prod.sh = run-tc-best.sh with its final exec pointed at run-tc-new.prod.sh
#   then execs run-tc-best.prod.sh with the prod overrides below in the environment. The originals are never edited.
#   If a substitution no longer matches (launcher restructured), it logs "TC-PROD WARN" and still starts (worst case:
#   default cache dir / memguard log path); if the best->new redirect fails it execs the ORIGINAL base unchanged.
# Prod overrides (env, applied on top of run-tc-best.sh's defaults):
#   KVOFF_GB=16 (forced; TC_PROD_KVOFF_GB=N to change; tcctl's injected KVOFF_GB=0 is ignored on purpose)
#   MAX_MODEL_LEN=262144, PROF=0, port 8030 + served names ornith-122b qwen38-27b qwen36-27b (from run-tc-new.sh, checked)
#   VLLM_CACHE_ROOT=$TC_CACHE_ROOT (default /opt/1cat/tcflash-trial/cache-tcprod)
#   MEMGUARD_LOG=/opt/1cat/ornith122-trial/logs/memguard.log (svc_health_watchdog's MG_LOG), memguard metrics/health :8030
#   --scheduling-policy priority + max-num-seqs 4 (from run-tc-new.sh, checked) = priority-client lane / router reserved slot
#   WEXT=tc_prod_ext.TCProdExt wrapping the base's WEXT (adds the PR #1232 RAM-cache fix, ext/tc_upfix_pr1232.py)
#   TC_SO_NODRAFT=1 -> --scheduler-cls tc_so_nodraft_sched.TCSONoDraftScheduler (R9 port; default 0, the JSON gate decides)
# Logs: engine output goes to the unit's log (ornith-prod.log) AND logs/prod-<date>.log (TC_PROD_TEE=0: unit log only).
#   One "TC-PROD start ..." line per start in logs/runs.log (live_facts reads kvoff= from it).
set -u
D=/opt/1cat/tcflash-trial
G=$D/golive/gen
mkdir -p "$G" "$D/logs"
TS=$(date +%Y%m%d-%H%M%S)
PLOG=$D/logs/prod-$TS.log
[ "${TC_PROD_DRY:-0}" = 1 ] && TC_PROD_TEE=0   # TC_PROD_DRY=1: write gen/ copies, print, do not start (safe, no GPU)
if [ "${TC_PROD_TEE:-1}" = 1 ]; then exec > >(tee -a "$PLOG") 2>&1; fi
say() { echo "$(date '+%F %T') TC-PROD $*"; }

# ---- prod overrides ----
export KVOFF_GB=${TC_PROD_KVOFF_GB:-16}
export MAX_MODEL_LEN=262144 PROF=0
export TC_CACHE_ROOT=${TC_CACHE_ROOT:-$D/cache-tcprod}
export TC_MEMGUARD_LOG=${TC_MEMGUARD_LOG:-/opt/1cat/ornith122-trial/logs/memguard.log}
export TC_UPFIX_PR1232=${TC_UPFIX_PR1232:-1}
TC_PROD_ARGS=""
[ "${TC_SO_NODRAFT:-0}" = 1 ] && TC_PROD_ARGS="--scheduler-cls tc_so_nodraft_sched.TCSONoDraftScheduler"
# 2026-10-27 TOOLFIX: chunking-proof qwen3_coder streaming (no-arg tools streamed "{" under load) + serving catch-up
# safety net. ext/tc_toolfix.py registers qwen3_coder_tcfix; the later --tool-call-parser overrides run-tc-new.sh's.
# Off: TC_TOOLFIX=0 (40-tc.conf). Offline test: ext/tests/test_tc_toolfix_cpu.py
if [ "${TC_TOOLFIX:-1}" = 1 ]; then
  TC_PROD_ARGS="$TC_PROD_ARGS --tool-parser-plugin $D/ext/tc_toolfix.py --tool-call-parser qwen3_coder_tcfix"
fi
export TC_PROD_ARGS

# ---- base launcher ----
BASE=$D/run-tc-best.sh; [ -f "$BASE" ] || { BASE=$D/run-tc-new.sh; say "WARN run-tc-best.sh missing -> base run-tc-new.sh (no best-known speed settings)"; }
NEW=$D/run-tc-new.sh
# worker extension the base would use -> wrapped by tc_prod_ext (unless the caller forces WEXT)
if [ -z "${WEXT:-}" ]; then
  BW=$(grep -oE 'WEXT=\$\{WEXT:-[A-Za-z0-9_.]+\}' "$BASE" | head -1 | sed -E 's/.*:-([^}]+)\}/\1/')
  [ -z "$BW" ] && BW=$(grep -oE 'EXT=\$\{WEXT:-[A-Za-z0-9_.]+\}' "$NEW" | head -1 | sed -E 's/.*:-([^}]+)\}/\1/')
  export TC_BASE_EXT=${TC_BASE_EXT:-${BW:-tc_ext.TCExt}}
  export WEXT=tc_prod_ext.TCProdExt
fi

# ---- derived copy of run-tc-new.sh ----
GN=$G/run-tc-new.prod.sh
sed -e 's#VLLM_CACHE_ROOT=/opt/1cat/tcflash-trial/cache-new#VLLM_CACHE_ROOT=${TC_CACHE_ROOT}#' \
    -e 's#MEMGUARD_LOG=/opt/1cat/tcflash-trial/logs/memguard.log#MEMGUARD_LOG=${TC_MEMGUARD_LOG}#' \
    -e 's#\${TC_EXTRA:-} --async-scheduling#${TC_EXTRA:-} ${TC_PROD_ARGS:-} --async-scheduling#' \
    "$NEW" > "$GN.tmp" && mv -f "$GN.tmp" "$GN"
grep -q 'VLLM_CACHE_ROOT=${TC_CACHE_ROOT}' "$GN" || say "WARN cache-root substitution missed (engine uses run-tc-new.sh's own cache dir)"
grep -q 'MEMGUARD_LOG=${TC_MEMGUARD_LOG}' "$GN" || say "WARN memguard-log substitution missed (svc_health_watchdog MG_LOG will not see memguard lines)"
grep -q 'TC_PROD_ARGS' "$GN" || { [ -n "$TC_PROD_ARGS" ] && say "WARN TC_PROD_ARGS ($TC_PROD_ARGS) could not be injected"; }
grep -q -- '--served-model-name ornith-122b qwen38-27b qwen36-27b' "$GN" || say "WARN served names changed in run-tc-new.sh"
grep -q -- '--port 8030' "$GN" || say "WARN port 8030 not found in run-tc-new.sh"
grep -q -- '--scheduling-policy priority' "$GN" || say "WARN --scheduling-policy priority not found in run-tc-new.sh (priority-client lane!)"
grep -q -- '--max-num-seqs 4' "$GN" || say "WARN --max-num-seqs 4 not found in run-tc-new.sh (router reserved-slot rule)"
grep -q -- '--max-model-len ${MAX_MODEL_LEN' "$GN" || say "WARN max-model-len no longer reads MAX_MODEL_LEN"

# ---- derived copy of the base ----
if [ "$BASE" = "$NEW" ]; then TARGET=$GN
else
  GB=$G/run-tc-best.prod.sh
  sed -e "s#exec bash $NEW#exec bash $GN#" "$BASE" > "$GB.tmp" && mv -f "$GB.tmp" "$GB"
  if grep -q "exec bash $GN" "$GB"; then TARGET=$GB
  else TARGET=$BASE; say "WARN could not redirect $BASE to the derived run-tc-new copy -> running the ORIGINAL base (cache-new + tcflash memguard log)"; fi
fi

say "start base=$(basename "$BASE") sha=$(sha256sum "$BASE" | cut -c1-12) new_sha=$(sha256sum "$NEW" | cut -c1-12) target=$TARGET wext=${WEXT:-} base_ext=${TC_BASE_EXT:-} kvoff=$KVOFF_GB cache=$TC_CACHE_ROOT so_nodraft=${TC_SO_NODRAFT:-0} pr1232=$TC_UPFIX_PR1232 toolfix=${TC_TOOLFIX:-1} log=$PLOG"
[ "${TC_PROD_DRY:-0}" = 1 ] && { say "DRY run: not starting"; exit 0; }
echo "$(date '+%F %T') TC-PROD start base=$(basename "$BASE") wext=${WEXT:-} base_ext=${TC_BASE_EXT:-} len=$MAX_MODEL_LEN kvoff=$KVOFF_GB cache=$TC_CACHE_ROOT so_nodraft=${TC_SO_NODRAFT:-0} log=$PLOG" >> $D/logs/runs.log
cd /tmp
exec bash "$TARGET"
