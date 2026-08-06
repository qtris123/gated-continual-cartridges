#!/usr/bin/env bash
# =====================================================================================
# REPRODUCE_ALL.sh — verify every headline number in the AM continual-cartridge
# investigation.  Generated 2026-07-30.
#
#   Usage:   bash research_loop/REPRODUCE_ALL.sh <target>
#   Targets: list | phase1 | s1 | s3 | s5 | s7 | zero_gpu | all
#            or a single experiment id, e.g.  bash ... DIAG-ROPE
#
# WHAT THIS SCRIPT DOES NOT DO
#   * It does not modify any source file.  Where a number can only be reproduced by
#     reverting a committed mechanism, the recipe is written as a comment marked
#     "### CODE CHANGE REQUIRED" and the arm is SKIPPED.  Four such cases exist (S3c).
#   * It does not commit.  It writes only under $OUT (default /tmp/repro_am).
#
# FIVE HAZARDS THIS SCRIPT ENCODES (each cost the original investigation time)
#   H1  eval_forgetting.py PRINTS `Eval loss` then HANGS forever.  Never `wait` on it —
#       parse the line, then kill the PID.  eval_ckpt() below does this.
#   H2  Unpinned `import cartridges` on this box resolves to the SIBLING repo
#       /localhome/local-triv/gated-continual-cartridges, which lacks rope_theta,
#       key_reposition and the MECH-008/009 selectors.  We PYTHONPATH-pin and verify.
#   H3  The verification probe is a FALSE NEGATIVE from the repo root: for `python -c`,
#       cwd is sys.path[0].  Probe from /tmp.  (verify_import() below does.)
#   H4  Never `env VAR=val bash ...` — a ~/.local/bin/env PATH shim swallows it
#       silently (rc=0, no-op).  Use `VAR=val bash ...` or export.
#   H5  One job per GPU via flock.  Two training jobs on one device = OOM + garbage
#       timings.  claim_gpu() below blocks until a device is free.
#
# TWO EVAL CONVENTIONS — do not mix them
#   in-run     : the loss printed by the training job itself (phase2_summary.json)
#   standalone : eval_forgetting.py re-run on a saved checkpoint
#   DIAG-NOISE measured these differ by up to 0.0065 on a BYTE-IDENTICAL cartridge.
#   Every EXPECT below is tagged [inrun] or [standalone].  results.csv mixes both.
#
# TOLERANCES
#   The closed-form solve is DETERMINISTIC — six nominally identical runs produced
#   byte-identical caches.  So a reproduction should match to ~1e-6, not to the eval
#   noise band.  TOL_EXACT is therefore tight.  The ±0.049 MT / ±0.054 QA figure is the
#   measured PAIRED resolution for COMPARING TWO ARMS — it is not a reproduction
#   tolerance.  Do not use it as one.
# =====================================================================================
set -uo pipefail

REPO=/localhome/local-triv/gated-continual-cartridges_explore
OUT=${OUT:-/tmp/repro_am}
SNAP=${SNAP:-/tmp/repro_am_snapshot}
TOL_EXACT=${TOL_EXACT:-0.0005}     # reproduction tolerance (solve is deterministic)
PHASE1=outputs/phase1_selfdistill_qwen512/cache_last.pt
MT_DATA=data/qasper/train/qwen_qasper_MT_task_8192.parquet
EVAL_QA=data/qasper/eval/qasper_eval_QA.parquet
EVAL_MT=data/qasper/eval/qasper_eval_MT.parquet

mkdir -p "$OUT"
cd "$REPO" || exit 9
export CARTRIDGES_DIR=$PWD
export CARTRIDGES_OUTPUT_DIR=$PWD/outputs
PY="$CARTRIDGES_DIR/.venv/bin/python"

# ---------------------------------------------------------------- helpers ----------
banner(){ echo; echo "############ $* ############"; }

pin_snapshot(){   # H2: freeze committed code so a concurrent edit cannot contaminate
  rm -rf "$SNAP"; mkdir -p "$SNAP"
  git -C "$REPO" archive HEAD cartridges examples | tar -x -C "$SNAP"
  export PYTHONPATH="$SNAP:${PYTHONPATH:-}"
  echo "PINNED_TO=$SNAP  HEAD=$(git -C "$REPO" rev-parse --short HEAD)"
}
verify_import(){  # H3: MUST be run from /tmp, not the repo root
  local p; p=$(cd /tmp && "$PY" -c "import cartridges,os;print(os.path.dirname(cartridges.__file__))")
  echo "IMPORT_PATH=$p"
  case "$p" in "$SNAP"/*|"$REPO"/*) ;; *) echo "FATAL: resolved to sibling repo (H2)"; exit 2;; esac
}
claim_gpu(){      # H5
  local d=/tmp/gpu_locks_${USER:-$(id -un)}; mkdir -p "$d"
  for _ in $(seq 1 240); do
    for i in 1 0; do
      exec {fd}>"$d/gpu${i}.lock"
      if flock -n "$fd"; then export CUDA_VISIBLE_DEVICES=$i MASTER_PORT=$((29500+i));
         echo "CLAIMED_GPU=$i"; return 0; fi
      exec {fd}>&-
    done; sleep 15
  done; echo "NO_FREE_GPU"; return 3
}

# run_am <logfile>  — env for the arm must already be exported by the caller
run_am(){
  local log="$1"
  bash examples/qasper2/scripts/core/train_continual_am_sparse.sh >"$log" 2>&1
  local rc=$?
  local dir; dir=$(grep -oE "Saved to /localhome/\S+" "$log" | tail -1 | awk '{print $3}')
  echo "RC=$rc RUNDIR=$dir"
  echo "$dir"
}

# eval_ckpt <ckpt> <eval_parquet> <tag>  — H1: parse the line, then KILL
eval_ckpt(){
  # `local a=1 b=$a` cannot be collapsed: bash expands every word of the `local`
  # command before performing any assignment, so $tag would be unbound under `set -u`.
  local ckpt="$1" data="$2" tag="$3"
  local log="$OUT/eval_${tag}.log"
  CHECKPOINT_PATH="$ckpt" EVAL_DATA_PATH="$data" RUN_NAME="REPRO_${tag}" \
    "$PY" examples/qasper2/train/eval_forgetting.py >"$log" 2>&1 &
  local pid=$!
  # The line reads "... - INFO - Eval loss - <x>" and is flushed onto the tail of a
  # tqdm progress line, so it can be neither anchored at ^ nor split on a colon.
  for _ in $(seq 1 600); do grep -qE 'Eval loss' "$log" && break; sleep 1; done
  sleep 1                                   # let the number finish flushing
  local loss; loss=$(grep -oE 'Eval loss[^0-9]*[0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+$')
  kill -TERM "$pid" 2>/dev/null; sleep 2; kill -KILL "$pid" 2>/dev/null   # H1
  grep -q "$ckpt" "$log" || echo "WARN: intended ckpt not confirmed in $log"
  echo "$loss"
}

# expect <label> <measured> <target> [tol]
expect(){
  local lab="$1" got="$2" want="$3" tol="${4:-$TOL_EXACT}"
  "$PY" - "$lab" "$got" "$want" "$tol" <<'EOF'
import sys
lab,got,want,tol=sys.argv[1],sys.argv[2],sys.argv[3],float(sys.argv[4])
if not got: print(f"[MISSING] {lab}: no value parsed (expected {want})"); sys.exit(0)
d=abs(float(got)-float(want))
print(f"[{'OK ' if d<=tol else 'FAIL'}] {lab}: got {got}  expect {want}  |d|={d:.6f} tol={tol}")
EOF
}

# k-curve snapshot helper: k documents written -> cache-after-doc-<k-1>
snap_k(){ ls "$1"/cache-after-doc-$(printf '%03d' $(( $2 - 1 )))-*.pt 2>/dev/null | head -1; }

# common canonical env (theta=1e4 default == the historical/buggy value: see S3a)
canon(){
  export PHASE1_CACHE_PATH=$PHASE1 SYNTH_DATA_PATH=$MT_DATA
  export TOP_T=32 GRANULARITY=per_layer SLOT_SELECTION=tfidf USE_IDF=0
  export KEY_MODE=freeze
  export RIDGE_LAMBDA=1e-4 RIDGE_SCALE=spectral DELTA_WEIGHT=1e-2
  export MAX_QUERIES_PER_HEAD=64 ENABLE_BETA=0
  unset AM_ROPE_THETA AM_KEY_REPOSITION AM_SEED_OFFSET AM_SAFE_FRACTION AM_SAFE_METRIC \
        AM_ORACLE_WRITE AM_ONPOLICY_LAYERS 2>/dev/null || true
}

# ============================== PHASE-1 FLOOR (gate everything) =====================
r_phase1(){ banner "PHASE-1 FLOOR — the anchor every other number is relative to"
  claim_gpu || return 3
  local qa mt
  qa=$(eval_ckpt "$PHASE1" "$EVAL_QA" phase1_QA); mt=$(eval_ckpt "$PHASE1" "$EVAL_MT" phase1_MT)
  expect "Phase-1 QA [standalone]" "$qa" 2.23880672454834
  expect "Phase-1 MT [standalone]" "$mt" 3.7825491428375244
}

# ============================== S1: is the method wired up? ========================
r_diag_wire(){ banner "DIAG-WIRE — target_mode never reached the per-document write (now structural)"
  # PURPOSE: EXP-008 swept TARGET_MODE over 3 values and got BIT-IDENTICAL results.
  #          Three different targets cannot give one solution -> wiring bug, not a null.
  # STATUS:  RESOLVED BY DELETION, not by a fix. The AM package restructure removed
  #          `target_mode` entirely: the per-document path always builds the
  #          [cartridge || doc] teacher, which is what the field pretended to select.
  #          The 3-arm sweep can no longer demonstrate anything -- an unknown env var
  #          is simply ignored, so the three arms would agree for the WRONG reason.
  #          Re-run it at a pre-restructure commit if you need the original artefact.
  #          See research_loop/AM_CONFIG.md "Deleted".
  echo "SKIPPED: target_mode no longer exists (see AM_CONFIG.md 'Deleted')."
  echo "The finding is now encoded in the code, not reproducible as a sweep."
  return 0
}

r_oracle_write(){ banner "ORACLE-WRITE — the write-ceiling oracle (MECH-001)"
  # PURPOSE: bound the ENTIRE value-only family in one run by writing the teacher's own
  #          document KV into the selected slots instead of solving.  If a PERFECT value
  #          write cannot reach the bar, no better solve can.
  claim_gpu || return 3; canon
  export AM_ORACLE_WRITE=1 RUN_NAME=REPRO_ORACLE-WRITE
  local d; d=$(run_am "$OUT/oracle.log" | tail -1)
  expect "ORACLE-WRITE QA [standalone]" "$(eval_ckpt "$d/cache_last.pt" "$EVAL_QA" ow_QA)" 1.8955 0.002
  expect "ORACLE-WRITE MT [standalone]" "$(eval_ckpt "$d/cache_last.pt" "$EVAL_MT" ow_MT)" 2.3810 0.002
  echo "INTERPRETATION: a perfect value write reaches only MT 2.381 vs the 1.8725 bar."
}

# ============================== S3: the three bugs =================================
r_diag_rope(){ banner "S3a  DIAG-ROPE — the 500x rotary-base error"
  # PURPOSE: cartridges/am/ hard-codes rope_theta=10000.0 in EVERY entry point
  #          (core.py:24,51,83,122; teacher.py:142,164,187; key_select.py:59,110,236)
  #          and NO CALLER PASSES IT.  The model is Qwen3-4B-Instruct-2507 -> 5000000.
  #          At T_doc 3858-8900 the two rotations are decorrelated (E[cos]=0.00 at 4k);
  #          76.6% of the 64 frequency pairs are off by more than pi.
  # NOTE: arm A needs NO code change — theta=1e4 is still the DEFAULT (MECH-003 kept the
  #       historical value so old runs reproduce).  Arm B just sets the flag.
  claim_gpu || return 3
  canon; RUN_NAME=REPRO_DIAG-ROPE-A; local dA; dA=$(run_am "$OUT/ropeA.log" | tail -1)
  canon; export AM_ROPE_THETA=5000000; RUN_NAME=REPRO_DIAG-ROPE-B
  local dB; dB=$(run_am "$OUT/ropeB.log" | tail -1)
  expect "ROPE-A QA [standalone]" "$(eval_ckpt "$dA/cache_last.pt" "$EVAL_QA" ropeA_QA)" 2.17662 0.002
  expect "ROPE-A MT [standalone]" "$(eval_ckpt "$dA/cache_last.pt" "$EVAL_MT" ropeA_MT)" 2.54836 0.002
  expect "ROPE-B QA [standalone]" "$(eval_ckpt "$dB/cache_last.pt" "$EVAL_QA" ropeB_QA)" 2.15320 0.002
  expect "ROPE-B MT [standalone]" "$(eval_ckpt "$dB/cache_last.pt" "$EVAL_MT" ropeB_MT)" 2.52961 0.002
  echo "--- THE POINT IS NOT THE CE (dMT -0.019, inside noise). CHECK THE INTERNALS: ---"
  for t in A B; do d=$([ $t = A ] && echo "$dA" || echo "$dB")
    echo "arm $t: $("$PY" -c "
import json;s=json.load(open('$d/phase2_summary.json'))
print('mean_mse',s.get('am_mean_mse'),'|v|max',s.get('value_global_max_abs'))" 2>/dev/null)"; done
  echo "EXPECT mean am/mean_mse 0.11906 -> 0.01406 (8.5x more fittable); |v|max 984 -> 178."
  echo "ALSO: layers 34/35 held 93-95% of residual and collapsed 26x/138x -> that"
  echo "      'binding constraint' was a rotary artefact."
}

r_mech_queries(){ banner "S3b  MECH-QUERIES / -B — query count is NOT a lever"
  # PURPOSE: max_queries_per_head was hard-coded 64 (finetune.py:81) with no knob, vs
  #          the paper's 16k-50k.  The accumulator can supply 57,344-81,920 per KV-head
  #          per document, so ~99.9% was being discarded.  MECH-002 exposes it.
  # THE CONFOUND (found by the worker against its own result): the guarded solve stacks
  #          [X_new (n x t) ; sqrt(w) I (t x t)], so the trust region's relative pull
  #          decays like 1/n.  Sweeping n at fixed DELTA_WEIGHT moves TWO things.
  #          MECH-QUERIES-B fixes it with w = 1e-2 * n/64.
  claim_gpu || return 3
  echo "--- A: fixed DELTA_WEIGHT=1e-2 (the confounded sweep) ---"
  for n in 64 256 1024 4096 16384; do
    canon; export MAX_QUERIES_PER_HEAD=$n RUN_NAME="REPRO_MECH-QUERIES_n$n"
    local d; d=$(run_am "$OUT/mq_$n.log" | tail -1)
    echo "n=$n QA=$(eval_ckpt "$d/cache_last.pt" "$EVAL_QA" mq${n}_QA) MT=$(eval_ckpt "$d/cache_last.pt" "$EVAL_MT" mq${n}_MT)"
  done
  echo "EXPECT [standalone] QA/MT:  64 -> 2.1772/2.5524 | 256 -> 2.4097/2.7716"
  echo "                          1024 -> 2.2575/2.5744 | 4096 -> 2.3412/2.7114"
  echo "                         16384 -> 2.3325/2.6943      (best MT is the CONTROL)"
  echo "                         |v|max should GROW 984 -> 7808, the wrong direction."
  echo "--- B: DELTA_WEIGHT scaled with n (confound removed) ---"
  for pair in "1024 0.16" "16384 2.56"; do set -- $pair
    canon; export MAX_QUERIES_PER_HEAD=$1 DELTA_WEIGHT=$2 RUN_NAME="REPRO_MECH-QUERIES-B_n$1"
    local d; d=$(run_am "$OUT/mqb_$1.log" | tail -1)
    echo "n=$1 w=$2 QA=$(eval_ckpt "$d/cache_last.pt" "$EVAL_QA" mqb$1_QA) MT=$(eval_ckpt "$d/cache_last.pt" "$EVAL_MT" mqb$1_MT)"
  done
  echo "EXPECT [standalone] 1024/0.16 -> 2.0401/2.5986 ; 16384/2.56 -> 2.0263/2.6084"
  echo "  MT +0.046/+0.056 = INSIDE noise and on the WRONG side. |v|max 7808 -> 656."
  echo "  QA falls to 2.0263, i.e. 0.21 BELOW the Phase-1 floor 2.2388."
  echo "  Write is NOT trivial: 50.3% Frobenius displacement, 1974 slots changed vs 1960."
  echo "  Cost is flat: 256x the queries for 1.23x wall clock."
}

r_mech_beta(){ banner "S3c  MECH-BETA — beta runs clean, and makes BOTH axes worse"
  # PURPOSE: beta is AM's own mass-matching mechanism and had failed 3 times
  #          (EXP-005 cholesky not-PD, EXP-005b lstsq NaN, EXP-006 NaN inside the fit).
  # ROOT CAUSES (LIT-002, all fixed by MECH-004):
  #   (i)   key_select.py:35 used torch.linalg.lstsq with the DEFAULT `gels` driver,
  #         which on CUDA returns NaN/Inf for rank-deficient input WITHOUT RAISING, so
  #         the `except RuntimeError` at :36 never fired.  -> AM_NNLS_DRIVER=gelsd.
  #   (ii)  no upper bound anywhere; the 1e-12 floor put beta at -27.6 vs the paper's
  #         box [-3,3] with iters=2.  -> AM_BETA_BOX=3.0, AM_NNLS_ITERS=2.
  #   (iii) a residual-target clamp absent from the paper.  -> AM_BETA_TARGET.
  #   (iv)  _should_fit_beta silently DISABLED beta whenever KEY_MODE=freeze, so beta
  #         had never run by construction.  -> decoupled by MECH-004.
  claim_gpu || return 3
  canon; export AM_ROPE_THETA=5000000 RUN_NAME=REPRO_MECH-BETA_rope
  local dR; dR=$(run_am "$OUT/beta_rope.log" | tail -1)
  canon; export AM_ROPE_THETA=5000000 ENABLE_BETA=1 AM_BETA_BOX=3.0 AM_NNLS_ITERS=2 \
                AM_NNLS_DRIVER=gelsd RUN_NAME=REPRO_MECH-BETA_on
  local dB; dB=$(run_am "$OUT/beta_on.log" | tail -1)
  expect "beta-off QA [standalone]" "$(eval_ckpt "$dR/cache_last.pt" "$EVAL_QA" bR_QA)" 2.15320 0.002
  expect "beta-off MT [standalone]" "$(eval_ckpt "$dR/cache_last.pt" "$EVAL_MT" bR_MT)" 2.52961 0.002
  expect "beta-ON  QA [standalone]" "$(eval_ckpt "$dB/cache_last.pt" "$EVAL_QA" bB_QA)" 2.73637 0.002
  expect "beta-ON  MT [standalone]" "$(eval_ckpt "$dB/cache_last.pt" "$EVAL_MT" bB_MT)" 2.81193 0.002
  echo "EXPECT mass_on_S (MT) 0.0711 -> 0.3010 = 4.23x, yet MT/QA ratio only 1.043 -> 1.051."
  echo "REASON: beta_j is a PER-SLOT CONSTANT in softmax(q.k_j/sqrt(d) + beta_j), so it"
  echo "  shifts every query's logit equally -> maps MT and QA mass through the SAME"
  echo "  monotone function.  It moves BANDWIDTH, never SELECTIVITY.  Predicted before"
  echo "  the run: +2.32 nats takes mass 0.090->0.500 but the ratio only 1.083->1.046."
  echo "  59.7% of beta entries pin at the +3 ceiling."
  #
  # ### CODE CHANGE REQUIRED (4 arms NOT run by this script) ###
  # To reproduce the ORIGINAL failures you must revert MECH-004 — do it on a scratch
  # branch, never on trivo-explore-research-work:
  #   (a) EXP-005  cholesky not-PD  : restore the unclamped NNLS (no AM_BETA_BOX path)
  #                                   in cartridges/am/key_select.py::refit_beta_nnls
  #                                   and run ENABLE_BETA=1 RIDGE_LAMBDA=1e-4.
  #   (b) EXP-005b lstsq NaN        : same, with RIDGE_LAMBDA=0.
  #   (c) EXP-006  NaN inside fit   : restore the `gels` default at key_select.py:35
  #                                   (AM_NNLS_DRIVER=gels may reproduce it via the flag —
  #                                   try that FIRST, it needs no edit — but the box and
  #                                   the residual-target clamp are also part of MECH-004).
  #   (d) "beta silently off"       : restore the old _should_fit_beta in finetune.py
  #                                   (~L264-269) so it returns key_mode != "freeze",
  #                                   then run ENABLE_BETA unset with KEY_MODE=freeze and
  #                                   confirm beta never fits.
  # Also NOT reproducible: "mass_on_S went unrecorded for nine experiments" — MECH-002
  # added the emission from the DELTA_WEIGHT branch, so today it is always logged.
}

# ============================== S5: keys — the one confirmed positive ==============
r_mech_keys(){ banner "S5  MECH-KEYS (MECH-005) — the project's ONE confirmed positive"
  # PURPOSE: every row in results.csv was KEY_MODE=freeze; "keys collapse QA" was
  #          pre-loop, Llama-era folklore and had NEVER been tested here.
  # HAZARD H2': key installation puts a DOCUMENT key (prefilled at position 512+m) into
  #          a CARTRIDGE slot (position 0..511) with no counter-rotation.  MECH-005 adds
  #          it.  Unit check: repositioned logit error 4.8e-07 vs 1.054 uncorrected — and
  #          doing the same rotation at theta=1e4 gives 4.4-5.1, WORSE than not
  #          correcting, which is why the driver refuses the flag without AM_ROPE_THETA.
  claim_gpu || return 3
  canon; export AM_ROPE_THETA=5000000 KEY_MODE=highest_attention AM_KEY_REPOSITION=1 \
                ENABLE_BETA=0 RUN_NAME=REPRO_MECH-KEYS_repos
  local d; d=$(run_am "$OUT/keys.log" | tail -1)
  expect "keys+repos k=16 QA [standalone]" "$(eval_ckpt "$(snap_k "$d" 16)" "$EVAL_QA" k16_QA)" 2.034916400909424
  expect "keys+repos k=16 MT [standalone]" "$(eval_ckpt "$(snap_k "$d" 16)" "$EVAL_MT" k16_MT)" 2.330503463745117
  expect "keys+repos k=12 QA [standalone]" "$(eval_ckpt "$(snap_k "$d" 12)" "$EVAL_QA" k12_QA)" 1.9560121297836304
  expect "keys+repos k=12 MT [standalone]" "$(eval_ckpt "$(snap_k "$d" 12)" "$EVAL_MT" k12_MT)" 2.2720184326171875
  echo "k=12 is the BEST point of the run, NOT the k=16 endpoint. Compare against the"
  echo "frozen control's own optimum (k=10, MT 2.4083) -> dMT -0.1363; seed-varied mean"
  echo "over 3 offsets at each arm's per-seed argmin = -0.1545 (2.8-3.6x resolution)."
  echo "QA side is NOT confirmed: consistently negative, consistently inside the band."
}

r_mech_seed(){ banner "S6  MECH-SEED (MECH-007) — seed variation was IMPOSSIBLE before this"
  # PURPOSE: the per-document draw is seeded with the CONSTANT doc_idx, and `config.seed`
  #          never reaches it, so `seed=N` on argv does NOT change the draw.  (Before the
  #          restructure this was compounded by pydrantic.main being reached only in the
  #          dead AM_EXECUTION_MODE=train_loop; the runner now always goes through it, but
  #          `config.seed` still does not reach the draw -- AM_SEED_OFFSET is the knob.)
  #          => every number predating MECH-007 is ONE DRAW, including the prior loop's
  #          "confirmed" win.  AM_SEED_OFFSET replaces ~30 of each doc's 32 reference
  #          conversations (Jaccard 0.036-0.043; 0 of 16 docs keep their draw).
  claim_gpu || return 3
  for off in 0 1000 2000; do
    canon; export AM_ROPE_THETA=5000000 KEY_MODE=highest_attention AM_KEY_REPOSITION=1 \
                  AM_SEED_OFFSET=$off RUN_NAME="REPRO_MECH-SEED_off$off"
    local d; d=$(run_am "$OUT/seed_$off.log" | tail -1)
    echo "off=$off k12 MT=$(eval_ckpt "$(snap_k "$d" 12)" "$EVAL_MT" seed${off}_MT)"
  done
  echo "EXPECT k=12 MT 2.27202 / 2.26954 / 2.26263 — across-seed range only 0.0094."
  echo "AM_SEED_OFFSET=0 must be BIT-IDENTICAL to the flag being unset (17 artefacts)."
}

# ============================== S7: the gating hypothesis ==========================
r_infogate(){ banner "S7a  MECH-INFOGATE (MECH-008) — every information-theoretic selector loses"
  # PURPOSE: test statistical top-t selection properly.  Precondition WAS validated:
  #          Spearman(tf_mass_qa, fisher) = 0.579 -> attention mass is NOT importance,
  #          so the family was not bounded a priori.
  # NOT IMPLEMENTED ON PURPOSE: kl_loo.  The exact leave-one-out KL is -log(1-w_j), a
  #          strictly monotone function of the slot's own attention weight (measured
  #          rho = 0.968 with mass).  Ranking by LOO-KL IS ranking by attention mass.
  # NOT BUILT (acknowledged gap): an `entropy` SELECTOR arm.  entropy was scored and
  #          correlated (rho 0.237 with Fisher, near-orthogonal) but never ranked with.
  claim_gpu || return 3
  local base_ok=1
  for arm in "tfidf 32" "redundancy 32" "redundancy 64" "redundancy 128" "mass_x_redundancy 32" "fisher 32"; do
    set -- $arm
    canon; export AM_ROPE_THETA=5000000 KEY_MODE=highest_attention AM_KEY_REPOSITION=1 \
                  SLOT_SELECTION=$1 TOP_T=$2 RUN_NAME="REPRO_MECH-INFOGATE_$1_t$2"
    local d; d=$(run_am "$OUT/ig_$1_$2.log" | tail -1)
    for k in 8 10 12 16; do
      echo "$1 t=$2 k=$k QA=$(eval_ckpt "$(snap_k "$d" $k)" "$EVAL_QA" ig_$1_$2_${k}_QA) MT=$(eval_ckpt "$(snap_k "$d" $k)" "$EVAL_MT" ig_$1_$2_${k}_MT)"
    done
  done
  cat <<'TXT'
EXPECT (best MT at each arm's own optimum k, [standalone]):
  control tfidf t32        MT 2.2720   QA 1.9464   realised MT routing mass 0.2799
  redundancy t32           MT 2.4386  (+0.167)     mass 0.0662
  redundancy t64           MT 2.4811  (+0.209)     mass 0.1195
  redundancy t128          MT 2.4582  (+0.186)     mass 0.2146   <- 4x budget, STILL less
  mass_x_redundancy t32    MT 2.3789  (+0.107)     mass 0.1251        mass than incumbent
  fisher t32               MT 2.6692  (+0.397)  QA 1.8641  mass 0.0104
KEY RELATION: log(MT routing mass) -> best MT, Pearson -0.877 across these six arms.
  The incumbent MAXIMISES MT routing mass BY CONSTRUCTION, so any importance-aware
  selector captures less of it and necessarily loses.
fisher = the anti-alignment end to end: BEST retention ever measured here, +0.397 MT.
CAVEAT (later audit): `redundancy` uses the WRONG Gram — the correct damage-of-
  overwriting is ||v_j||^2 / (H^-1)_jj with H = E_q[a a^T] (routing second moment, the
  OBS/SparseGPT saliency).  Ours used G = V V^T (value space) and DIVIDED ||v_j||^2 out.
  G is query-independent, so `redundancy` cannot be a QA-importance metric at all.
  Also: "redundancy is the best gradient-free Fisher proxy (rho -0.648)" is an
  ECOLOGICAL correlation — within-layer it is -0.323.  Fixing both DEEPENS the
  anti-alignment from 11x to 48x below chance.  See notes/2026-07-29-lit-rev/04-*.
TXT
}

r_budget(){ banner "S7b  MECH-BUDGET / -B — support is not a lever (and my confound)"
  # PURPOSE: MECH-INFOGATE's only control was top_t=32, which confounds "bigger budget
  #          helps" with "smarter selection helps".  MECH-BUDGET supplies plain tfidf at
  #          larger budgets so the attribution is possible.
  # MY ERROR: I pinned MAX_QUERIES_PER_HEAD=64 in the base config, so t128 solved 64
  #          equations for 128 unknowns per head — UNDER-DETERMINED, the exact flaw the
  #          brief cited as invalidating HYP-S1.  MECH-BUDGET-B removes it.
  # CORRECTION to my mechanism claim: I said t128 fell into the solver's min-norm branch
  #          (core.py:198-204).  Measured: with DELTA_WEIGHT>0 the stacked design is
  #          (n+t) x t, so that branch is NEVER taken.  The underdetermination is in the
  #          DATA BLOCK — rank 64, leaving a 64-dim null space per head pinned to its
  #          prior at |dv|/|v| = 0.0004.  Half of every head's support carried no new
  #          information, which is why that arm had the LOWEST am/mean_mse.
  claim_gpu || return 3
  echo "--- A: confounded (queries pinned at 64) ---"
  for t in 64 128; do
    canon; export AM_ROPE_THETA=5000000 KEY_MODE=highest_attention AM_KEY_REPOSITION=1 \
                  TOP_T=$t RUN_NAME="REPRO_MECH-BUDGET_t$t"
    local d; d=$(run_am "$OUT/bud_$t.log" | tail -1)
    for k in 8 12; do echo "t=$t k=$k MT=$(eval_ckpt "$(snap_k "$d" $k)" "$EVAL_MT" bud${t}_${k}_MT)"; done
  done
  echo "EXPECT t64 best MT 2.2318 (k=8); t128 best 2.3571 (k=12) <- CONFOUNDED, see above"
  echo "--- B: determined (queries raised; n >> t) ---"
  for pair in "128 512 1e-2" "64 512 1e-2" "256 1024 1e-2" "128 512 8e-2"; do set -- $pair
    canon; export AM_ROPE_THETA=5000000 KEY_MODE=highest_attention AM_KEY_REPOSITION=1 \
                  TOP_T=$1 MAX_QUERIES_PER_HEAD=$2 DELTA_WEIGHT=$3 \
                  RUN_NAME="REPRO_MECH-BUDGET-B_t$1_q$2_w$3"
    local d; d=$(run_am "$OUT/budb_$1_$2_$3.log" | tail -1)
    echo "t=$1 q=$2 w=$3 best-of-k MT: $(for k in 8 10 12 16; do eval_ckpt "$(snap_k "$d" $k)" "$EVAL_MT" budb$1$2$3_${k}; done | sort -g | head -1)"
  done
  cat <<'TXT'
EXPECT best MT: t128/q512 2.4075 | t64/q512 2.3768 | t256/q1024 2.3860 | t128/q512/w8e-2 2.3595
THE CLEAN CONTRAST (matched n=512, matched w, BOTH determined) t128 vs t64:
  dMT +0.0346 / +0.0284 / -0.0230 / -0.0185 at k=8/10/12/16 — ALL inside the +-0.049
  paired resolution AND NOT SIGN-CONSISTENT.  Support is not a lever.
THE CONTROL FIRES: at t64, already determined at n=64, raising n ALONE to 512 makes
  BOTH axes worse at every k (MT +0.130..+0.172) — query count is not a free change,
  so only matched-n contrasts isolate support.
TXT
}

r_constrained(){ banner "S7c  MECH-CONSTRAINED (MECH-009) — the dose-response that closes it"
  # PURPOSE: the STRONGEST form of the hypothesis — mass-ranked WITHIN a safety
  #          constraint, which is the selector DIAG-IMPORTANCE's projection described
  #          (MECH-INFOGATE tested PURE top-t-by-redundancy, 2.4-3.2x below it).
  # BUILD CHECK: q=1.0 must give IDENTICAL index AND score tensors to attention_mass.
  claim_gpu || return 3
  for pair in "0.75 32" "0.50 32" "0.25 32" "0.50 64"; do set -- $pair
    canon; export AM_ROPE_THETA=5000000 KEY_MODE=highest_attention AM_KEY_REPOSITION=1 \
                  SLOT_SELECTION=constrained_mass AM_SAFE_FRACTION=$1 AM_SAFE_METRIC=redundancy \
                  TOP_T=$2 RUN_NAME="REPRO_MECH-CONSTRAINED_q$1_t$2"
    local d; d=$(run_am "$OUT/cons_$1_$2.log" | tail -1)
    echo "q=$1 t=$2 best-of-k MT: $(for k in 8 10 12 16; do eval_ckpt "$(snap_k "$d" $k)" "$EVAL_MT" cons$1$2_${k}; done | sort -g | head -1)"
  done
  cat <<'TXT'
EXPECT (best MT / realised MT routing mass), TOP_T fixed at 32, only q moving:
  q=1.00 (=incumbent) 2.2720 / 0.2799      <- the curve's OPTIMUM is the incumbent
  q=0.75              2.3288 / 0.1661
  q=0.50              2.3693 / 0.1327
  q=0.25              2.4388 / 0.1000
  q=0.50 @ t64        2.3999 / 0.1842      <- bigger budget does NOT rescue it
MONOTONE in both columns.  Pearson log(mass) vs MT = -0.9775 within the sweep,
-0.8681 pooled over 10 arms with MECH-INFOGATE.  Every dMT clears +-0.049; EVERY QA
gain sits INSIDE +-0.054.  There is no untried q where this might have worked.
SECOND FINDING: offline doc-0 projections are ~0.6x OPTIMISTIC (realised/predicted
0.749/0.598/0.567/0.585/0.612) because redundancy is recomputed on the DRIFTING live
cache while the written union grows.  Carry that factor for any static extrapolation.
TXT
}

r_verify_gate(){ banner "S7d  VERIFY-GATE — the gating negative survives seed variation"
  claim_gpu || return 3
  for off in 0 1000 2000; do
    for sel in redundancy tfidf; do
      canon; export AM_ROPE_THETA=5000000 KEY_MODE=highest_attention AM_KEY_REPOSITION=1 \
                    SLOT_SELECTION=$sel AM_SEED_OFFSET=$off RUN_NAME="REPRO_VERIFY-GATE_${sel}_off$off"
      local d; d=$(run_am "$OUT/vg_${sel}_$off.log" | tail -1)
      echo "$sel off=$off best-of-k MT: $(for k in 8 10 12 16; do eval_ckpt "$(snap_k "$d" $k)" "$EVAL_MT" vg${sel}${off}_${k}; done | sort -g | head -1)"
    done
  done
  cat <<'TXT'
EXPECT dMT (redundancy - tfidf) at each arm's own per-seed argmin:
  off 0    2.43865 (k16) vs 2.27202 (k12)  = +0.16663  (3.37x resolution)
  off 1000 2.43268 (k10) vs 2.26954 (k12)  = +0.16314  (3.30x)
  off 2000 2.46260 (k16) vs 2.26263 (k12)  = +0.19997  (4.05x)
ALL 12 matched-k cells clear even the CONSERVATIVE composite +-0.1055, and so does the
adversarial pairing (redundancy's best value at ANY seed vs tfidf's worst): +0.16066.
log(MT routing mass) -> best MT gives Pearson -0.986 ACROSS SEEDS — so the relation is
not an artefact of which selectors were compared.
QA: -0.0217 / -0.0486 / -0.0251 — consistently negative, NO offset clears +-0.054.
NOTE: this negative is BETTER ESTABLISHED than the project's own positive, whose QA
axis does not clear the composite.
TXT
}

# ============================== zero-GPU re-analyses ================================
r_zero_gpu(){ banner "ZERO-GPU — re-analyses that can change the conclusions"
  cat <<'TXT'
These need no training and mostly no GPU.  Each already has a worker script on disk;
run them against the arrays in research_loop/state/diagnostics/.

 N1  swap the Gram      : recompute `redundancy` with H = E_q[a a^T] (routing second
                          moment) instead of G = V V^T, and re-derive the trade-off
                          table.  Fixes the error above; MECH-009's dose-response must
                          then be re-read as a dose over the CORRECT metric.
                          Start from research_loop/results/DIAG-IMPORTANCE/measure_slot_importance.py
 N2  free Fisher proxy  : E[w^2] ~= 2*(kl_loo - w_mass) from arrays already on disk.
                          EXPECT per-layer rho with Fisher 0.847 vs tf_mass 0.666 and
                          redundancy -0.323; Fisher-safest-32 agreement 0.753.
 N3  IDF forensic       : read mass_on_S off an archived USE_IDF=1 run.  Thread 02
                          derives that USE_IDF=1 is `constrained_mass` WITH THE
                          CONSTRAINT INVERTED (IDF_TOP_K=128 + rho~0.958 rankings =>
                          idf ~ 0 on the HIGHEST-mass slots => deletion).
                          PREDICT realised mass ~= 0.014.
                          FALSIFIER: mass >= 0.10 while MT ~ 2.73 => a second axis
                          exists and the whole gating family reopens.
 N4  PMI_DC re-scoring  : calibrate against the matched-domain control in
                          research_loop/state/diagnostics/DIAG-CONTENT.json.
                          Canonical-minus-control widens 0.232/0.288/0.608/0.882 at
                          k=4/8/12/16 => k=16 is the best point, NOT k=12; the tail
                          "regression" is largely content-free.
 N5  effective rank of E_q[a a^T] : seconds.  Could close the family by a capacity
                          theorem.  Reuse research_loop/results/DIAG-KEYSPACE/measure_query_geometry.py
 --  DIAG-NOISE re-check: research_loop/results/DIAG-NOISE/analyze.py recovers
                          per-example losses EXACTLY (the metric is a token-weighted
                          micro-average, so bucketing ce_by_token by element_ids
                          reproduces 92 published evals to <= 4.0e-07).
                          EXPECT paired 95% interval +-0.049 MT / +-0.054 QA;
                          independent +-0.256/+-0.236; single arm +-0.186/+-0.177.
TXT
}

# ============================== dispatcher =========================================
case "${1:-list}" in
  list) sed -n '1,40p' "$0"; echo "targets: phase1 s1 s3 s5 s7 zero_gpu all | DIAG-WIRE ORACLE-WRITE DIAG-ROPE MECH-QUERIES MECH-BETA MECH-KEYS MECH-SEED MECH-INFOGATE MECH-BUDGET MECH-CONSTRAINED VERIFY-GATE";;
  phase1) pin_snapshot; verify_import; r_phase1;;
  s1) pin_snapshot; verify_import; r_diag_wire; r_oracle_write;;
  s3) pin_snapshot; verify_import; r_diag_rope; r_mech_queries; r_mech_beta;;
  s5) pin_snapshot; verify_import; r_mech_keys; r_mech_seed;;
  s7) pin_snapshot; verify_import; r_infogate; r_budget; r_constrained; r_verify_gate;;
  zero_gpu) r_zero_gpu;;
  all) pin_snapshot; verify_import; r_phase1; r_diag_wire; r_oracle_write; r_diag_rope
       r_mech_queries; r_mech_beta; r_mech_keys; r_mech_seed
       r_infogate; r_budget; r_constrained; r_verify_gate; r_zero_gpu;;
  DIAG-WIRE) pin_snapshot; verify_import; r_diag_wire;;
  ORACLE-WRITE) pin_snapshot; verify_import; r_oracle_write;;
  DIAG-ROPE) pin_snapshot; verify_import; r_diag_rope;;
  MECH-QUERIES) pin_snapshot; verify_import; r_mech_queries;;
  MECH-BETA) pin_snapshot; verify_import; r_mech_beta;;
  MECH-KEYS) pin_snapshot; verify_import; r_mech_keys;;
  MECH-SEED) pin_snapshot; verify_import; r_mech_seed;;
  MECH-INFOGATE) pin_snapshot; verify_import; r_infogate;;
  MECH-BUDGET) pin_snapshot; verify_import; r_budget;;
  MECH-CONSTRAINED) pin_snapshot; verify_import; r_constrained;;
  VERIFY-GATE) pin_snapshot; verify_import; r_verify_gate;;
  *) echo "unknown target: $1"; exit 1;;
esac
