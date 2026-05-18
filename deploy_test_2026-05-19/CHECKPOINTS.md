# Inventaire des checkpoints — deploy test 2026-05-19

23 checkpoints, prêts à déployer un par un sur le vrai SO-101 (Eval1 + Eval2,
familles 16×16 px ET 64×64 px). Chemin = nom du fichier (tout est encodé
dedans). `infer.py` auto-détecte taille image + Eval1/2 depuis les poids —
même commande pour tous, on change juste `--checkpoint`.

```
RAPPELS
- 16px : conv kernel 4 (2-conv)   | 64px : conv kernel 8 (3-conv)
- n_state 12 = Eval1 (1 cube, pas de couleur)
  n_state 18 = Eval2 (2 cubes, couleur cible -> --goal_color)
- TOUS env NATIVE (reward/evaluate/terminations jamais touchés ;
  seule "criterion" = Eval2 succès conditionné couleur, env séparé).
  Seule variable entre étapes = DR knobs + image size + warm-start.
- success_at_end rapporté NON FIABLE 2 sens (loophole hover / cube hors bol)
  -> les % ci-dessous = comptage VISUEL /50 quand il existe, sinon "non
  compté". Le run robot réel reste le seul vrai juge.
```

---

## EVAL1 — `SO101PlaceCube-v1`, n_state=12 (1 cube dans le bol, pas de couleur)

### 16×16 px — premier livrable (campagne autonome 17/05)
```
checkpoints/eval1/16px/
  eval1_16px_S1_base_noDR.pt
     base : tâche native + actionneur fidèle 10Hz Feetech, DR OFF
     succès visuel : non compté /50 (étape intermédiaire)
  eval1_16px_S2_DRctrl.pt
     + DR dynamique (stiffness/damping/delay/lag)
     succès visuel : non compté /50
  eval1_16px_S3_DRvisu.pt
     + DR visuel (lighting/caméra)
     succès visuel : non compté /50
  eval1_16px_S4_DRfull_6color.pt
     + 6 couleurs (= env HEAD, DR complète)
     succès visuel : ~62% (31/50)
  eval1_16px_DELIVERED_warmS4.pt           <<< LE LIVRÉ 16px
     warm(S1) -> env S4 (DR complète + 6 coul.), 3M
     md5 0fbbcecfee4e07f3e78c3cfe36560ff2
     succès : ~70-78% genuine (40/50 métrique + 13 frames full-res ;
              PAS un /50 complet)
     !! Sur le VRAI robot avant correction mapping : ÉCHEC false-grasp.
        -> teste-le AVEC --mapping correct (et compare à --mapping native).
```

### 64×64 px — curriculum warm-start (18/05) — `eval1_curriculum_handoff`
```
checkpoints/eval1/64px/
  eval1_64px_S1_base_noDR.pt
     cold 4M, DR OFF, no-jitter, no-exposure (env pur natif)
     succès visuel : ~26/50 (52%)  [compté main]
  eval1_64px_S2_DRctrl_BEST.pt             <<< MEILLEUR Eval1 (recommandé)
     warm S1 + DR contrôle (lighting off), 3M
     succès visuel : ~33/50 (66%)  [compté main] — réduit aussi le loophole
  eval1_64px_S3_DRvisu_expo_COLLAPSE.pt
     warm S2 + DR visuel+jitter+exposure (agressif d'un coup), 3M
     succès : COLLAPSE training (success 0.00, ckpt jamais amélioré)
```

---

## EVAL2 — `SO101PlaceCubeEval2-v1`, n_state=18 (2 cubes, --goal_color requis)

`--goal_color` : 0 red · 1 blue · 2 green · 3 yellow · 4 purple · 5 orange.
Mettre la couleur du cube à ranger ; l'autre cube (distracteur) = autre couleur.

### 16×16 px — premier livrable, échelle B (tous comptés /50)
```
checkpoints/eval2/16px/
  eval2_16px_base_noDR.pt
     DR OFF
     ~92% MAIS non déployable (0 robustesse)
  eval2_16px_B1_DRctrl.pt
     + DR dynamique (visuel OFF)
     74% (37/50)
  eval2_16px_B15_DRctrl_cam.pt
     + bruit caméra (lighting OFF)
     74% (37/50) — robuste caméra
  eval2_16px_B2_DRfull.pt
     + lighting agressif (full)
     64% (32/50) — régression
  eval2_16px_B2p_DRfull_agg.pt
     + lighting agressif (depuis B1.5)
     62% (31/50)
  eval2_16px_DELIVERED_B175.pt             <<< LE LIVRÉ 16px Eval2
     warm B1.5 + lighting réaliste étroit (ambient 0.28-0.42, dir 0.85-1.15)
     md5 a67cd585384c3b4cf03e038ae47d8461
     68% (34/50) — meilleur compromis DÉPLOYABLE
```

### 64×64 px — curriculum FIN 9 étapes (18/05) — `eval2_curriculum_handoff`
```
checkpoints/eval2/64px/
  eval2_64px_S1_base_noDR.pt     cold 4M, DR OFF
  eval2_64px_S2_ctrl_mild.pt     + DR ctrl MILD (1.5M, idem ↓)
  eval2_64px_S3_ctrl_full.pt     + DR ctrl FULL
  eval2_64px_S4_light_mild.pt    + lighting MILD
  eval2_64px_S5_light_full.pt    + lighting FULL      } sweet-spot
  eval2_64px_S6_jitter.pt        + color jitter       } typique
  eval2_64px_S7_expo030.pt       + exposure x0.30     } ~ S5-S7
  eval2_64px_S8_expo060.pt       + exposure x0.60
  eval2_64px_S9_expo100.pt       + exposure x1.00 (max enveloppe réduite)

  Plusieurs ckpt_best fins restés au warm-init (métrique non améliorée
  sous la DR ajoutée) -> meilleure policy propagée, AUCUNE dégradation.
  Pas de collapse brutal (rampe douce). Succès non recompté -> juger
  à l'oeil sur le robot ; sweet-spot attendu ~ S5-S7.
```

---

## Ordre de test conseillé — CHAQUE checkpoint ×2 MAPPINGS

> Règle : pour tout checkpoint testé, faire **2 runs dos à dos** :
> `--mapping correct` PUIS `--mapping native` (même ckpt, même scène).
> C'est le but du jour : comparer le mapping validé vs le natif squint
> sur le VRAI robot. Priorité décroissante ci-dessous ; idéalement les
> 23 ckpts ×2 = 46 runs, sinon au moins les candidats prioritaires ×2.

```
EVAL1   (chaque ligne = 2 runs : correct puis native)
  1. eval1_64px_S2_DRctrl_BEST.pt        (top candidat, 66% sim)
  2. eval1_16px_DELIVERED_warmS4.pt      (1er livrable)
  3. eval1_64px_S1_base_noDR.pt
  4. eval1_16px_S4_DRfull_6color.pt
  5. reste 16px S1/S2/S3 + 64px S3(collapse, attendu mauvais)

EVAL2   (chaque ligne = 2 runs ; toujours --goal_color <couleur cible>)
  1. eval2_16px_DELIVERED_B175.pt        (déployable, 68% sim)
  2. eval2_64px_S6_jitter.pt / S5 / S7   (sweet-spot ~S5-S7)
  3. eval2_16px_B15_DRctrl_cam.pt
  4. reste 16px (B1/base/B2/B2p) + 64px (S1/S2/S3/S4/S8/S9)
```

Note tes observations dans `RESULTS.md` (gabarit fourni) : ce que TU vois
faire au robot (genuine = cube vraiment dans le bol, pince ouverte/retirée,
pas de hover), pas une métrique.
