#!/usr/bin/env bash
# Eval2 FINE graded curriculum (user: many more stages + gentler exposure).
# Reuses the already-running eval2_cur_s1 (base, no DR) as the warm root,
# then 8 fine warm-chained stages ramping DR gradually; exposure introduced
# LATE and GENTLY via EXPO_DR_SCALE (0.30 -> 0.60 -> 1.00 of the already-
# reduced envelope). DR knobs only — reward/evaluate/env stay NATIVE.
# Detached, compaction-proof; logs to runs_logs/CURRICULUM_STATE.md.
set +e
cd /home/rayane-sebti/Desktop/MA2/squint-native-iso
PY=/home/rayane-sebti/miniconda3/envs/squint/bin/python
ST=runs_logs/CURRICULUM_STATE.md
GATE=$HOME/Desktop/curriculum_gates
ENV=SO101PlaceCubeEval2-v1
mkdir -p "$GATE" runs_logs
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$ST"; }
COMMON="--obs-mode rgb --image-size 64 --render-size 128 --num-envs 512 --buffer-size 200000 --num-updates 128"

# wait for the running base (eval2_cur_s1) to finish
log "EVAL2-FINE: waiting for base eval2_cur_s1 to finish..."
while true; do
  if [ -f runs/eval2_cur_s1/ckpt_best.pt ] && ! pgrep -fa "train_squint.py .*eval2_cur_s1" >/dev/null 2>&1; then
    log "EVAL2-FINE: base eval2_cur_s1 done -> starting fine chain"; break
  fi
  sleep 60
done
# gate the base too (for the human planche set)
gate(){ local exp="$1"; local ck="runs/$exp/ckpt_best.pt"; [ -f "$ck" ]||ck="runs/$exp/ckpt.pt";
  [ -f "$ck" ]||{ log "GATE SKIP $exp (no ckpt)"; return; }
  log "GATE PLANCHE $exp"
  $PY view_policy.py --ckpt "$ck" --demos 50 --render rgb_array --env-id "$ENV" \
     --obs-mode rgb --image-size 64 --save > "runs_logs/${exp}_gate.log" 2>&1
  local nw; nw=$(ls -t "$HOME"/Desktop/*_planche.png 2>/dev/null|head -1)
  [ -n "$nw" ] && cp "$nw" "$GATE/${exp}_planche.png" && log "GATE READY $exp -> $GATE/${exp}_planche.png" || log "GATE WARN $exp no planche"; }

gate eval2_cur_s1

# stage <exp> <warm_ckpt> <ts> <EXPO_SCALE|-> <extra flags...>
stage(){ local exp="$1" warm="$2" ts="$3" es="$4"; shift 4; local extra="$*"
  if [ -f "runs/$exp/ckpt_best.pt" ] && grep -q "DONE $exp " "$ST" 2>/dev/null; then log "SKIP $exp (done)"; return; fi
  local ckf=""; if [ -f "$warm" ]; then ckf="--checkpoint $warm"; elif [ -f "${warm/ckpt_best/ckpt}" ]; then ckf="--checkpoint ${warm/ckpt_best/ckpt}"; log "WARN $exp warm from ckpt.pt"; else log "WARN $exp NO warm -> cold"; fi
  local pre=""; [ "$es" != "-" ] && pre="EXPO_DR_SCALE=$es"
  log "TRAIN START $exp ts=$ts expo='${es}' warm='${ckf:-cold}' extra='$extra'"
  env $pre $PY train_squint.py --env-id "$ENV" --exp-name "$exp" --total-timesteps "$ts" \
      $COMMON $ckf $extra > "runs_logs/${exp}.log" 2>&1
  if [ -f "runs/$exp/ckpt_best.pt" ] || [ -f "runs/$exp/ckpt.pt" ]; then
    log "TRAIN END $exp"; log "DONE $exp $(date '+%s')"; gate "$exp"
  else log "TRAIN FAIL $exp (tail:)"; tail -n 4 "runs_logs/${exp}.log"|sed 's/^/    /'|tee -a "$ST"; fi; }

TS=1500000
B=runs/eval2_cur_s1/ckpt_best.pt
stage eval2f_s2 "$B"                       $TS - --env-domain-randomization --no-apply-jitter --no-exposure-dr --dr-config-json runs_logs/dr_s2_ctrl_mild.json
stage eval2f_s3 runs/eval2f_s2/ckpt_best.pt $TS - --env-domain-randomization --no-apply-jitter --no-exposure-dr --dr-config-json runs_logs/dr_ctrl.json
stage eval2f_s4 runs/eval2f_s3/ckpt_best.pt $TS - --env-domain-randomization --no-apply-jitter --no-exposure-dr --dr-config-json runs_logs/dr_s4_light_mild.json
stage eval2f_s5 runs/eval2f_s4/ckpt_best.pt $TS - --env-domain-randomization --no-apply-jitter --no-exposure-dr
stage eval2f_s6 runs/eval2f_s5/ckpt_best.pt $TS - --env-domain-randomization --apply-jitter --no-exposure-dr
stage eval2f_s7 runs/eval2f_s6/ckpt_best.pt $TS 0.30 --env-domain-randomization --apply-jitter --exposure-dr
stage eval2f_s8 runs/eval2f_s7/ckpt_best.pt $TS 0.60 --env-domain-randomization --apply-jitter --exposure-dr
stage eval2f_s9 runs/eval2f_s8/ckpt_best.pt $TS 1.00 --env-domain-randomization --apply-jitter --exposure-dr
log "================ EVAL2 FINE CHAIN COMPLETE ================"
log "Planches: eval2_cur_s1 + eval2f_s2..s9 in $GATE (human /50). Build eval2 handoff next."
