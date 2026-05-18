[05-18 04:55:22] ================ CURRICULUM RUN START ================
[05-18 04:55:22] Eval1 = SO101PlaceCube-v1 ; Eval2 = SO101PlaceCubeEval2-v1
[05-18 04:55:22] TRAIN START eval1_cur_s1 env=SO101PlaceCube-v1 ts=4000000 warm='cold' extra='--no-env-domain-randomization --no-apply-jitter --no-exposure-dr'
[05-18 05:49:22] TRAIN END eval1_cur_s1 rc=0 (ckpt present)
[05-18 05:49:22] DONE eval1_cur_s1 1779076162
[05-18 05:49:22] GATE PLANCHE eval1_cur_s1 (50 demos, clean env, ckpt=runs/eval1_cur_s1/ckpt_best.pt)
[05-18 05:49:52] GATE READY eval1_cur_s1 -> /home/rayane-sebti/Desktop/curriculum_gates/eval1_cur_s1_planche.png  (HUMAN: count successes /50)
[05-18 05:49:52] TRAIN START eval1_cur_s2 env=SO101PlaceCube-v1 ts=3000000 warm='--checkpoint runs/eval1_cur_s1/ckpt_best.pt' extra='--env-domain-randomization --no-apply-jitter --no-exposure-dr --dr-config-json runs_logs/dr_ctrl.json'

[GATE VISUEL HUMAIN] eval1_cur_s1 : genuine ~26/50 (52%) — ECHEC seuil 38/50.
  Metrique rapportee 0.88 surevaluee. Cause = placement imparfait + LOOPHOLE
  critere succes (hover cube-en-pince + cube-hors-bol comptes True : #7,12,19,
  26,27,40,47,48). DR non en cause (S1 sans DR). Fix loophole = reward/evaluate
  => sanction user requise (conflit keep-native). Pipeline continue (S2/S3).
  Zips construits sur meilleur stage HONNETE, DEPLOY.md veridique. NE PAS
  modifier reward/evaluate sans sanction. NE PAS livrer zip "75%" sur <38/50.
[05-18 06:30:42] TRAIN END eval1_cur_s2 rc=0 (ckpt present)
[05-18 06:30:42] DONE eval1_cur_s2 1779078642
[05-18 06:30:42] GATE PLANCHE eval1_cur_s2 (50 demos, clean env, ckpt=runs/eval1_cur_s2/ckpt_best.pt)
[05-18 06:31:29] GATE READY eval1_cur_s2 -> /home/rayane-sebti/Desktop/curriculum_gates/eval1_cur_s2_planche.png  (HUMAN: count successes /50)
[05-18 06:31:29] TRAIN START eval1_cur_s3 env=SO101PlaceCube-v1 ts=3000000 warm='--checkpoint runs/eval1_cur_s2/ckpt_best.pt' extra='--env-domain-randomization --apply-jitter --exposure-dr'

[GATE VISUEL HUMAIN] eval1_cur_s2 : genuine ~33/50 (66%) — sous seuil 75%
  mais NETTE amelioration vs S1 (26/50). Metrique rapportee 0.44 = SOUS-compte
  (S1 0.88 sur-comptait) -> metrique non fiable 2 sens, seul visuel /50 vaut.
  Echecs S2 surtout rates honnetes (cube hors bol m=F), loophole hover reduit.
  warm+DR-controle ameliore genuine ET reduit loophole. S3 (warm S2 +visu
  +exposure) en cours -> gate ensuite. Si S3 <38/50 : best stage honnete =
  S2(66%) ; piste loosen DR-controle + rechaine si on vise 75%.
[05-18 07:13:22] TRAIN END eval1_cur_s3 rc=0 (ckpt present)
[05-18 07:13:22] DONE eval1_cur_s3 1779081202
[05-18 07:13:22] GATE PLANCHE eval1_cur_s3 (50 demos, clean env, ckpt=runs/eval1_cur_s3/ckpt_best.pt)
[05-18 07:14:10] GATE READY eval1_cur_s3 -> /home/rayane-sebti/Desktop/curriculum_gates/eval1_cur_s3_planche.png  (HUMAN: count successes /50)
[05-18 07:14:10] EVAL CHAIN COMPLETE eval1_cur (planches ready for human /50 gate: s1/s2/s3)
[05-18 07:14:10] TRAIN START eval2_cur_s1 env=SO101PlaceCubeEval2-v1 ts=4000000 warm='cold' extra='--no-env-domain-randomization --no-apply-jitter --no-exposure-dr'

[EVAL1 HANDOFF LIVRE] ~/Desktop/eval1_curriculum_handoff.zip (S1+S2+S3 ckpt,
  infer_eval1_64.py mapping valide 64px archi OK strict, 3 planches, DEPLOY
  honnete). S3 = COLLAPSE training (success 0.00 return 3.6, ckpt jamais
  ameliore) sous DR visu+exposure agressive -> S2 = best (~66% visuel). User
  consulte les planches lui-meme (je ne recompte plus /50). Eval2 en cours.

[EVAL2 REDESIGN user 2026-05-18] Ancien plan eval2 3-stage ABANDONNE.
  Nouveau: curriculum FIN 9 etapes (eval2_cur_s1 base RUNNING garde +
  eval2f_s2..s9), DR graduee mild->full, EXPOSURE late+gentle via
  EXPO_DR_SCALE 0.30/0.60/1.00 sur enveloppe ExposureDRWrapper REDUITE
  (0.7-1.5/.07/.85-1.25/.05 ; scale0=neutre). DR knobs only, keep-native.
  Orchestrateur _eval2_fine.sh (detached) attend fin eval2_cur_s1 puis chaine.
[05-18 07:23:37] EVAL2-FINE: waiting for base eval2_cur_s1 to finish...
[05-18 08:09:38] EVAL2-FINE: base eval2_cur_s1 done -> starting fine chain
[05-18 08:09:38] GATE PLANCHE eval2_cur_s1
[05-18 08:10:14] GATE READY eval2_cur_s1 -> /home/rayane-sebti/Desktop/curriculum_gates/eval2_cur_s1_planche.png
[05-18 08:10:14] TRAIN START eval2f_s2 ts=1500000 expo='-' warm='--checkpoint runs/eval2_cur_s1/ckpt_best.pt' extra='--env-domain-randomization --no-apply-jitter --no-exposure-dr --dr-config-json runs_logs/dr_s2_ctrl_mild.json'
[05-18 08:30:53] TRAIN END eval2f_s2
[05-18 08:30:53] DONE eval2f_s2 1779085853
[05-18 08:30:53] GATE PLANCHE eval2f_s2
[05-18 08:31:40] GATE READY eval2f_s2 -> /home/rayane-sebti/Desktop/curriculum_gates/eval2f_s2_planche.png
[05-18 08:31:40] TRAIN START eval2f_s3 ts=1500000 expo='-' warm='--checkpoint runs/eval2f_s2/ckpt_best.pt' extra='--env-domain-randomization --no-apply-jitter --no-exposure-dr --dr-config-json runs_logs/dr_ctrl.json'
[05-18 08:52:18] TRAIN END eval2f_s3
[05-18 08:52:18] DONE eval2f_s3 1779087138
[05-18 08:52:18] GATE PLANCHE eval2f_s3
[05-18 08:53:14] GATE READY eval2f_s3 -> /home/rayane-sebti/Desktop/curriculum_gates/eval2f_s3_planche.png
[05-18 08:53:14] TRAIN START eval2f_s4 ts=1500000 expo='-' warm='--checkpoint runs/eval2f_s3/ckpt_best.pt' extra='--env-domain-randomization --no-apply-jitter --no-exposure-dr --dr-config-json runs_logs/dr_s4_light_mild.json'
[05-18 09:13:52] TRAIN END eval2f_s4
[05-18 09:13:52] DONE eval2f_s4 1779088432
[05-18 09:13:52] GATE PLANCHE eval2f_s4
[05-18 09:14:59] GATE READY eval2f_s4 -> /home/rayane-sebti/Desktop/curriculum_gates/eval2f_s4_planche.png
[05-18 09:14:59] TRAIN START eval2f_s5 ts=1500000 expo='-' warm='--checkpoint runs/eval2f_s4/ckpt_best.pt' extra='--env-domain-randomization --no-apply-jitter --no-exposure-dr'
[05-18 09:35:34] TRAIN END eval2f_s5
[05-18 09:35:34] DONE eval2f_s5 1779089734
[05-18 09:35:34] GATE PLANCHE eval2f_s5
[05-18 09:36:41] GATE READY eval2f_s5 -> /home/rayane-sebti/Desktop/curriculum_gates/eval2f_s5_planche.png
[05-18 09:36:41] TRAIN START eval2f_s6 ts=1500000 expo='-' warm='--checkpoint runs/eval2f_s5/ckpt_best.pt' extra='--env-domain-randomization --apply-jitter --no-exposure-dr'
[05-18 09:57:49] TRAIN END eval2f_s6
[05-18 09:57:49] DONE eval2f_s6 1779091069
[05-18 09:57:49] GATE PLANCHE eval2f_s6
[05-18 09:58:55] GATE READY eval2f_s6 -> /home/rayane-sebti/Desktop/curriculum_gates/eval2f_s6_planche.png
[05-18 09:58:55] TRAIN START eval2f_s7 ts=1500000 expo='0.30' warm='--checkpoint runs/eval2f_s6/ckpt_best.pt' extra='--env-domain-randomization --apply-jitter --exposure-dr'
[05-18 10:20:07] TRAIN END eval2f_s7
[05-18 10:20:07] DONE eval2f_s7 1779092407
[05-18 10:20:07] GATE PLANCHE eval2f_s7
[05-18 10:21:13] GATE READY eval2f_s7 -> /home/rayane-sebti/Desktop/curriculum_gates/eval2f_s7_planche.png
[05-18 10:21:13] TRAIN START eval2f_s8 ts=1500000 expo='0.60' warm='--checkpoint runs/eval2f_s7/ckpt_best.pt' extra='--env-domain-randomization --apply-jitter --exposure-dr'
[05-18 10:42:26] TRAIN END eval2f_s8
[05-18 10:42:26] DONE eval2f_s8 1779093746
[05-18 10:42:26] GATE PLANCHE eval2f_s8
[05-18 10:43:35] GATE READY eval2f_s8 -> /home/rayane-sebti/Desktop/curriculum_gates/eval2f_s8_planche.png
[05-18 10:43:35] TRAIN START eval2f_s9 ts=1500000 expo='1.00' warm='--checkpoint runs/eval2f_s8/ckpt_best.pt' extra='--env-domain-randomization --apply-jitter --exposure-dr'
[05-18 11:04:49] TRAIN END eval2f_s9
[05-18 11:04:49] DONE eval2f_s9 1779095089
[05-18 11:04:49] GATE PLANCHE eval2f_s9
[05-18 11:05:59] GATE READY eval2f_s9 -> /home/rayane-sebti/Desktop/curriculum_gates/eval2f_s9_planche.png
[05-18 11:05:59] ================ EVAL2 FINE CHAIN COMPLETE ================
[05-18 11:05:59] Planches: eval2_cur_s1 + eval2f_s2..s9 in /home/rayane-sebti/Desktop/curriculum_gates (human /50). Build eval2 handoff next.

[EVAL2 HANDOFF LIVRE 11:07] ~/Desktop/eval2_curriculum_handoff.zip (71M):
  9 ckpts S1..S9 + infer_eval2_64.py (strict-load verifie, mapping valide,
  n_state=18 color-cond) + 9 planches + DEPLOY honnete. Fait factuel:
  plusieurs ckpt_best fins = step0 (warm-init, metrique pas amelioree sous DR
  ajoutee) -> meilleure policy propagee (safe, pas de degradation). User juge
  succes sur planches (je ne recompte pas, sa consigne). Exposure douce
  graduee 0.30/0.60/1.00 sur enveloppe REDUITE -> pas de collapse brutal type
  S3-Eval1. ===== MANDAT AUTONOME COMPLET : Eval1 + Eval2 zips livres. =====
