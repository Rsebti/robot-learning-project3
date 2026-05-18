# Résultats robot réel — 2026-05-19

**RÈGLE : chaque checkpoint est testé avec LES DEUX mappings** (`correct`
puis `native`, dos à dos, même scène) → 2 lignes par checkpoint ci-dessous.
**Succès = jugé À L'ŒIL** : cube vraiment dans le bol, pince ouverte et
retirée, pas de hover, pas de cube hors table. Aucune métrique. `n/N` =
succès genuine / essais.

```
checkpoint                                  | mapping | goal | a_scale | n/N | observation
--------------------------------------------|---------|------|---------|-----|--------------------------
EVAL1
eval1_64px_S2_DRctrl_BEST.pt                | correct |  -   | 0.10    |  /  |
eval1_64px_S2_DRctrl_BEST.pt                | native  |  -   | 0.10    |  /  |
eval1_16px_DELIVERED_warmS4.pt              | correct |  -   | 0.10    |  /  |
eval1_16px_DELIVERED_warmS4.pt              | native  |  -   | 0.10    |  /  |
eval1_64px_S1_base_noDR.pt                  | correct |  -   | 0.10    |  /  |
eval1_64px_S1_base_noDR.pt                  | native  |  -   | 0.10    |  /  |
eval1_16px_S4_DRfull_6color.pt              | correct |  -   | 0.10    |  /  |
eval1_16px_S4_DRfull_6color.pt              | native  |  -   | 0.10    |  /  |
eval1_16px_S1_base_noDR.pt                  | correct |  -   | 0.10    |  /  |
eval1_16px_S1_base_noDR.pt                  | native  |  -   | 0.10    |  /  |
eval1_16px_S2_DRctrl.pt                     | correct |  -   | 0.10    |  /  |
eval1_16px_S2_DRctrl.pt                     | native  |  -   | 0.10    |  /  |
eval1_16px_S3_DRvisu.pt                     | correct |  -   | 0.10    |  /  |
eval1_16px_S3_DRvisu.pt                     | native  |  -   | 0.10    |  /  |
eval1_64px_S3_DRvisu_expo_COLLAPSE.pt       | correct |  -   | 0.10    |  /  |
eval1_64px_S3_DRvisu_expo_COLLAPSE.pt       | native  |  -   | 0.10    |  /  |
--------------------------------------------|---------|------|---------|-----|--------------------------
EVAL2  (goal = --goal_color : 0 red 1 blue 2 green 3 yellow 4 purple 5 orange)
eval2_16px_DELIVERED_B175.pt                | correct |  ?   | 0.10    |  /  |
eval2_16px_DELIVERED_B175.pt                | native  |  ?   | 0.10    |  /  |
eval2_64px_S6_jitter.pt                     | correct |  ?   | 0.10    |  /  |
eval2_64px_S6_jitter.pt                     | native  |  ?   | 0.10    |  /  |
eval2_64px_S5_light_full.pt                 | correct |  ?   | 0.10    |  /  |
eval2_64px_S5_light_full.pt                 | native  |  ?   | 0.10    |  /  |
eval2_64px_S7_expo030.pt                    | correct |  ?   | 0.10    |  /  |
eval2_64px_S7_expo030.pt                    | native  |  ?   | 0.10    |  /  |
eval2_16px_B15_DRctrl_cam.pt                | correct |  ?   | 0.10    |  /  |
eval2_16px_B15_DRctrl_cam.pt                | native  |  ?   | 0.10    |  /  |
eval2_16px_B1_DRctrl.pt                     | correct |  ?   | 0.10    |  /  |
eval2_16px_B1_DRctrl.pt                     | native  |  ?   | 0.10    |  /  |
eval2_16px_base_noDR.pt                     | correct |  ?   | 0.10    |  /  |
eval2_16px_base_noDR.pt                     | native  |  ?   | 0.10    |  /  |
eval2_16px_B2_DRfull.pt                     | correct |  ?   | 0.10    |  /  |
eval2_16px_B2_DRfull.pt                     | native  |  ?   | 0.10    |  /  |
eval2_16px_B2p_DRfull_agg.pt                | correct |  ?   | 0.10    |  /  |
eval2_16px_B2p_DRfull_agg.pt                | native  |  ?   | 0.10    |  /  |
eval2_64px_S1_base_noDR.pt                  | correct |  ?   | 0.10    |  /  |
eval2_64px_S1_base_noDR.pt                  | native  |  ?   | 0.10    |  /  |
eval2_64px_S2_ctrl_mild.pt                  | correct |  ?   | 0.10    |  /  |
eval2_64px_S2_ctrl_mild.pt                  | native  |  ?   | 0.10    |  /  |
eval2_64px_S3_ctrl_full.pt                  | correct |  ?   | 0.10    |  /  |
eval2_64px_S3_ctrl_full.pt                  | native  |  ?   | 0.10    |  /  |
eval2_64px_S4_light_mild.pt                 | correct |  ?   | 0.10    |  /  |
eval2_64px_S4_light_mild.pt                 | native  |  ?   | 0.10    |  /  |
eval2_64px_S8_expo060.pt                    | correct |  ?   | 0.10    |  /  |
eval2_64px_S8_expo060.pt                    | native  |  ?   | 0.10    |  /  |
eval2_64px_S9_expo100.pt                    | correct |  ?   | 0.10    |  /  |
eval2_64px_S9_expo100.pt                    | native  |  ?   | 0.10    |  /  |
```

## Conclusions du jour
- Meilleur Eval1 réel (ckpt + mapping) : …
- Meilleur Eval2 réel (ckpt + mapping) : …
- Mapping **correct** vs **native** — verdict A/B global : …
- À retenir pour la suite : …
