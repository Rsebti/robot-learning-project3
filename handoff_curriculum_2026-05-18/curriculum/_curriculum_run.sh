#!/usr/bin/env bash
# Autonomous curriculum chain warm-start : Eval1 then Eval2.
# Stages per eval: S1 base no-DR 64px (cold) -> S2 warm +DR-control
# -> S3 warm +DR-visu+exposure. Each stage: train to completion, then
# dump a 50-demo CLEAN-env planche for the human visual gate (/50, the
# metric is NOT trusted). Never stops; logs state to CURRICULUM_STATE.md.
# Resume-safe: stages whose ckpt_best.pt exists + marked DONE are skipped.
set +e
cd /home/rayane-sebti/Desktop/MA2/squint-native-iso
PY=/home/rayane-sebti/miniconda3/envs/squint/bin/python
ST=runs_logs/CURRICULUM_STATE.md
GATE=$HOME/Desktop/curriculum_gates
mkdir -p "$GATE" runs_logs
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$ST"; }

COMMON="--obs-mode rgb --image-size 64 --render-size 128 --num-envs 512 --buffer-size 200000 --num-updates 128"

# train_stage <env> <exp> <ckpt_or_-> <timesteps> <extra flags...>
train_stage(){
  local env="$1" exp="$2" ck="$3" ts="$4"; shift 4
  local extra="$*"
  if [ -f "runs/$exp/ckpt_best.pt" ] && grep -q "DONE $exp " "$ST" 2>/dev/null; then
    log "SKIP $exp (already DONE, ckpt_best.pt present)"; return 0
  fi
  local ckflag=""
  if [ "$ck" != "-" ]; then
    if [ -f "$ck" ]; then ckflag="--checkpoint $ck"
    elif [ -f "${ck/ckpt_best/ckpt}" ]; then ckflag="--checkpoint ${ck/ckpt_best/ckpt}"; log "WARN $exp: ckpt_best missing, warm from ckpt.pt"
    else log "WARN $exp: no warm ckpt ($ck) -> COLD"; fi
  fi
  log "TRAIN START $exp env=$env ts=$ts warm='${ckflag:-cold}' extra='$extra'"
  $PY train_squint.py --env-id "$env" --exp-name "$exp" --total-timesteps "$ts" \
      $COMMON $ckflag $extra > "runs_logs/${exp}.log" 2>&1
  local rc=$?
  if [ -f "runs/$exp/ckpt_best.pt" ] || [ -f "runs/$exp/ckpt.pt" ]; then
    log "TRAIN END $exp rc=$rc (ckpt present)"; log "DONE $exp $(date '+%s')"
  else
    log "TRAIN FAIL $exp rc=$rc (NO ckpt — see runs_logs/${exp}.log tail:)"
    tail -n 5 "runs_logs/${exp}.log" | sed 's/^/    /' | tee -a "$ST"
  fi
}

# gate_planche <env> <exp>  -> 50-demo CLEAN-env planche for human /50 count
gate_planche(){
  local env="$1" exp="$2"
  local ck="runs/$exp/ckpt_best.pt"; [ -f "$ck" ] || ck="runs/$exp/ckpt.pt"
  [ -f "$ck" ] || { log "GATE SKIP $exp (no ckpt)"; return 0; }
  log "GATE PLANCHE $exp (50 demos, clean env, ckpt=$ck)"
  $PY view_policy.py --ckpt "$ck" --demos 50 --render rgb_array \
      --env-id "$env" --obs-mode rgb --image-size 64 --save \
      > "runs_logs/${exp}_gate.log" 2>&1
  # view_policy writes ~/Desktop/<tag>_planche.png ; copy to a stable name
  local newest
  newest=$(ls -t "$HOME"/Desktop/*_planche.png 2>/dev/null | head -1)
  if [ -n "$newest" ]; then
    cp "$newest" "$GATE/${exp}_planche.png"
    log "GATE READY $exp -> $GATE/${exp}_planche.png  (HUMAN: count successes /50)"
  else
    log "GATE WARN $exp: no planche produced (tail:)"
    tail -n 5 "runs_logs/${exp}_gate.log" | sed 's/^/    /' | tee -a "$ST"
  fi
}

run_eval(){
  local env="$1" pfx="$2"
  train_stage "$env" "${pfx}_s1" "-" 4000000 \
      --no-env-domain-randomization --no-apply-jitter --no-exposure-dr
  gate_planche "$env" "${pfx}_s1"
  train_stage "$env" "${pfx}_s2" "runs/${pfx}_s1/ckpt_best.pt" 3000000 \
      --env-domain-randomization --no-apply-jitter --no-exposure-dr \
      --dr-config-json runs_logs/dr_ctrl.json
  gate_planche "$env" "${pfx}_s2"
  train_stage "$env" "${pfx}_s3" "runs/${pfx}_s2/ckpt_best.pt" 3000000 \
      --env-domain-randomization --apply-jitter --exposure-dr
  gate_planche "$env" "${pfx}_s3"
  log "EVAL CHAIN COMPLETE $pfx (planches ready for human /50 gate: s1/s2/s3)"
}

log "================ CURRICULUM RUN START ================"
log "Eval1 = SO101PlaceCube-v1 ; Eval2 = SO101PlaceCubeEval2-v1"
run_eval "SO101PlaceCube-v1"      "eval1_cur"
run_eval "SO101PlaceCubeEval2-v1" "eval2_cur"
log "================ CURRICULUM RUN ALL STAGES DONE ================"
log "NEXT (human): visual /50 gate each planche in $GATE ; >=38/50 -> build handoff zip from that stage's ckpt_best."
