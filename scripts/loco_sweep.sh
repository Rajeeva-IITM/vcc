#!/usr/bin/env bash
# Leave-one-context-out (LOCO) sweep: for each held-out context, train on the other 6
# (model_flow_film_kernel_linhead_qsp + dataset_multi_qsp_loco, reference_val_only), then
# score that context on the honest compass. Answers: does perturbation-direction transfer
# to an UNSEEN context? Reads DE-cos vs its own blind bar per context.
#
# Usage:  scripts/loco_sweep.sh
# Edit CONTEXTS / GPUS / EPOCHS below. One train runs per GPU at a time (contexts pipeline
# per GPU), so RAM stays at ~one 7-source datamodule per GPU. GPUS=(2) => fully sequential.
set -uo pipefail
cd "$(dirname "$0")/.."

# --- config -------------------------------------------------------------------
CONTEXTS=(replogle22_k562_preprocessed feng24_preprocessed nadig24_hepg2_preprocessed mcfaline23_gxe_preprocessed)
# Sequential: ONE GPU, contexts run one after another. Shared server -- do not hog GPUs.
# Add more indices (e.g. (0 1 3)) only if a run is time-critical and the GPUs are idle.
GPUS=(0)
EPOCHS=25
# Agnostic baseline by default. Context sweep: set these env vars on the invocation --
#   MODEL=model_flow_film_kernel_linhead_qsp_ctx PREFIX=loco_ctx \
#   EXTRA=data.datamodule.emit_context=true bash scripts/loco_sweep.sh
MODEL="${MODEL:-model_flow_film_kernel_linhead_qsp}"
PREFIX="${PREFIX:-loco}"
EXTRA="${EXTRA:-}"   # extra hydra overrides, space-separated
# ------------------------------------------------------------------------------

set -a; [ -f .env ] && source .env; set +a
RUN_DIR="${RUN_DIR:?RUN_DIR not set (check .env)}"
DATA_DIR="${DATA_DIR:?DATA_DIR not set (check .env)}"

h5ad() {  # source name -> its .h5ad path
  case "$1" in
    adata_2025_all) echo "$DATA_DIR/2025/adata_2025_all.h5ad" ;;
    *)              echo "$DATA_DIR/primeflow_data/$1.h5ad" ;;
  esac
}
best_ckpt() { ls "$RUN_DIR/$PREFIX/$1"/VCC-epoch=*.ckpt 2>/dev/null | head -1; }

train_one() {
  local ctx="$1" gpu="$2" dir="$RUN_DIR/$PREFIX/$1"
  if [ -n "$(best_ckpt "$ctx")" ]; then echo "[$ctx] ckpt exists, skip train"; return; fi
  mkdir -p "$dir"
  echo "[$ctx] train ($MODEL) on GPU $gpu -> $dir"
  pixi run python src/train.py \
    model="$MODEL" \
    data=dataset_multi_qsp_loco \
    data.datamodule.val_source="$ctx" \
    data.metadata.savename="$PREFIX/$ctx" \
    trainer.max_epochs="$EPOCHS" "trainer.devices=[$gpu]" \
    $EXTRA \
    > "$dir/train.log" 2>&1
}

score_one() {
  local ctx="$1" gpu="$2" dir="$RUN_DIR/$PREFIX/$1" ckpt
  ckpt="$(best_ckpt "$ctx")"
  if [ -z "$ckpt" ]; then echo "[$ctx] NO ckpt, skip score"; return; fi
  echo "[$ctx] score $ckpt on GPU $gpu"
  pixi run python scripts/score_local.py \
    --ckpt "$ckpt" --source "$(h5ad "$ctx")" \
    --model "$MODEL" \
    --data dataset_multi_qsp_loco \
    --device "cuda:$gpu" \
    > "$dir/score.log" 2>&1
}

# Assign contexts to GPUs round-robin; each GPU runs its contexts sequentially, GPUs parallel.
run_phase() {  # $1 = function name (train_one|score_one)
  local fn="$1" g ci
  for g in "${!GPUS[@]}"; do
    ( for ci in "${!CONTEXTS[@]}"; do
        [ $((ci % ${#GPUS[@]})) -eq "$g" ] && "$fn" "${CONTEXTS[$ci]}" "${GPUS[$g]}"
      done ) &
  done
  wait
}

echo "=== TRAIN phase ==="; run_phase train_one
echo "=== SCORE phase ==="; run_phase score_one

echo "=== AGGREGATE ==="
RUN_DIR="$RUN_DIR" PREFIX="$PREFIX" CONTEXTS="${CONTEXTS[*]}" pixi run python - <<'PY'
import os, json, glob, csv
run, ctxs, prefix = os.environ["RUN_DIR"], os.environ["CONTEXTS"].split(), os.environ["PREFIX"]
rows = []
for ctx in ctxs:
    mech = sorted(glob.glob(f"{run}/{prefix}/{ctx}/local_score_*/mechanism.json"))
    scd  = sorted(glob.glob(f"{run}/{prefix}/{ctx}/local_score_*/scored.csv"))
    if not mech:
        rows.append((ctx, None)); continue
    m = json.load(open(mech[-1]))
    overall = fid = nmae = float("nan")
    if scd:
        with open(scd[-1]) as f:
            r = list(csv.DictReader(f))
        col = "from_replicate" if r and "from_replicate" in r[0] else "from_baseline"
        by = {d["metric"]: d.get(col) for d in r}
        g = lambda k: float(by[k]) if by.get(k) not in (None, "") else float("nan")
        overall = g("avg_score")
        fid = g("de_wilcoxon_direction_fidelity_yield_raw")
        nmae = g("de_wilcoxon_lfc_nmae")
    rows.append((ctx, dict(de=m["de_cosine"], bar=m["de_cosine_blind"],
                           over=m["overshoot"], overall=overall, fid=fid, nmae=nmae)))

hdr = f"{'context':30s} {'DE-cos':>8s} {'blind bar':>9s} {'clears?':>7s} {'overshoot':>9s} {'overall':>8s} {'fid':>8s} {'nmae':>8s}"
print(hdr); print("-"*len(hdr))
out = [["context","de_cosine","blind_bar","clears","overshoot","overall","fid","nmae"]]
for ctx, d in rows:
    if d is None:
        print(f"{ctx:30s} {'--- no score ---':>44s}"); out.append([ctx]+[""]*7); continue
    clears = "YES" if d["de"] > d["bar"] else "no"
    print(f"{ctx:30s} {d['de']:>8.3f} {d['bar']:>9.3f} {clears:>7s} {d['over']:>9.2f} {d['overall']:>8.3f} {d['fid']:>8.3f} {d['nmae']:>8.3f}")
    out.append([ctx, f"{d['de']:.4f}", f"{d['bar']:.4f}", clears, f"{d['over']:.3f}", f"{d['overall']:.4f}", f"{d['fid']:.4f}", f"{d['nmae']:.4f}"])
p = f"{run}/{prefix}/loco_summary.csv"
csv.writer(open(p,"w")).writerows(out)
print(f"\nwrote {p}")
print("\nVERDICT: all 'no' => unseen-context direction unlearnable at this data scale (lever=data).")
print("         any 'YES' => context is the failure axis; build context-conditioning for those.")
PY
