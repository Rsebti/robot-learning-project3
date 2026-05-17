# Graph Report - /Users/admin/Documents/ETH/M4/Robot Learning /Project S101/robot-learning-project3/notes/visualizations  (2026-05-13)

## Corpus Check
- Corpus is ~3,887 words - fits in a single context window. You may not need a graph.

## Summary
- 50 nodes · 81 edges · 10 communities detected
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 3 edges (avg confidence: 0.73)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Visualization Code|Visualization Code]]
- [[_COMMUNITY_SmolVLA Finetune Pipeline|SmolVLA Finetune Pipeline]]
- [[_COMMUNITY_BC Method Alternatives|BC Method Alternatives]]
- [[_COMMUNITY_SmolVLA Architecture & Performance|SmolVLA Architecture & Performance]]
- [[_COMMUNITY_Robot Setup and Eval Tasks|Robot Setup and Eval Tasks]]
- [[_COMMUNITY_Dataset Schema|Dataset Schema]]
- [[_COMMUNITY_VisionAction Encoder Stack|Vision/Action Encoder Stack]]
- [[_COMMUNITY_Blackwell GPU Pin|Blackwell GPU Pin]]
- [[_COMMUNITY_Proprioception Path|Proprioception Path]]
- [[_COMMUNITY_Text Encoding Path|Text Encoding Path]]

## God Nodes (most connected - your core abstractions)
1. `SmolVLA` - 19 edges
2. `ACT` - 9 edges
3. `main()` - 7 edges
4. `101-episode Eval 2 dataset (projet3-eval2-v1-tom-hugo)` - 6 edges
5. `add_box()` - 5 edges
6. `Cross-Attention Transformer` - 5 edges
7. `Eval 2 (two-cube task)` - 5 edges
8. `Natural-language task string` - 5 edges
9. `add_arrow()` - 4 edges
10. `fig_smolvla_arch()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `ACT` --conceptually_related_to--> `Natural-language task string`  [INFERRED]
  notes/visualizations/smolvla_walkthrough.html → notes/visualizations/smolvla_walkthrough.html  _Bridges community 2 → community 5_
- `SmolVLA` --semantically_similar_to--> `ACT`  [EXTRACTED] [semantically similar]
  notes/visualizations/smolvla_walkthrough.html → notes/visualizations/smolvla_walkthrough.html  _Bridges community 3 → community 2_
- `SmolVLA` --references--> `Vision Encoder (SigLIP-style)`  [EXTRACTED]
  notes/visualizations/smolvla_walkthrough.html → notes/visualizations/smolvla_walkthrough.html  _Bridges community 3 → community 6_
- `SmolVLA` --references--> `Proprio MLP`  [EXTRACTED]
  notes/visualizations/smolvla_walkthrough.html → notes/visualizations/smolvla_walkthrough.html  _Bridges community 3 → community 8_
- `SmolVLA` --references--> `Text Encoder`  [EXTRACTED]
  notes/visualizations/smolvla_walkthrough.html → notes/visualizations/smolvla_walkthrough.html  _Bridges community 3 → community 9_

## Hyperedges (group relationships)
- **SmolVLA three input streams (image, proprio, text)** — vision_encoder, proprio_mlp, text_encoder, cross_attention [EXTRACTED 1.00]
- **Five-step finetune pipeline** — lerobot_smolvla_base, projet3_eval2_dataset, finetuning, brev_h100, deploy_infer_sh [EXTRACTED 1.00]
- **Named risks for SmolVLA finetune** — camera_mismatch_risk, wrist_only_underperformance, community_variance_risk, blackwell_sm120 [EXTRACTED 1.00]

## Communities

### Community 0 - "Visualization Code"
Cohesion: 0.4
Nodes (10): add_arrow(), add_box(), b64_png(), fig_act_vs_smolvla(), fig_finetune_pipeline(), fig_performance_numbers(), fig_smolvla_arch(), fig_two_cube_decision() (+2 more)

### Community 1 - "SmolVLA Finetune Pipeline"
Cohesion: 0.2
Nodes (10): Brev H100 finetune (~$8), Camera-name mismatch risk, deploy/infer.sh, Finetuning recipe (20k steps, batch 64, ~4h A100), Five-step finetune pipeline, lerobot/smolvla_base, POLICY_PATH runtime override, SmolVLA Pretraining (450M params) (+2 more)

### Community 2 - "BC Method Alternatives"
Cohesion: 0.4
Nodes (6): ACT, 50% target-color obedience ceiling, FiLM-conditioned ACT, HIL-SERL, Image augmentation, lerobot framework

### Community 3 - "SmolVLA Architecture & Performance"
Cohesion: 0.5
Nodes (5): Community reproduction variance (LeRobot issue #2915), Expected 70-90% in-distribution success, SmolVLA, SmolVLA paper (Shukor et al. 2026, arXiv:2506.01844), Vision-Language-Action architecture

### Community 4 - "Robot Setup and Eval Tasks"
Cohesion: 0.5
Nodes (4): Eval 1 (single-cube task), Eval 2 (two-cube task), SO-101 robot, Wrist camera (RGB)

### Community 5 - "Dataset Schema"
Cohesion: 0.5
Nodes (4): Bowl position (-15.5,29.5) cm, 101-episode Eval 2 dataset (projet3-eval2-v1-tom-hugo), single_task metadata field, Natural-language task string

### Community 6 - "Vision/Action Encoder Stack"
Cohesion: 0.5
Nodes (4): Action Head (joint target chunks), Cross-Attention Transformer, SigLIP, Vision Encoder (SigLIP-style)

### Community 7 - "Blackwell GPU Pin"
Cohesion: 1.0
Nodes (2): Blackwell sm_120 (RTX 5090) wheel mismatch, torch==2.9.1+cu128 pin

### Community 8 - "Proprioception Path"
Cohesion: 1.0
Nodes (2): Proprio MLP, Proprioception (joint angles)

### Community 9 - "Text Encoding Path"
Cohesion: 1.0
Nodes (2): SmolLM-base, Text Encoder

## Knowledge Gaps
- **13 isolated node(s):** `Rounded box with centered text.`, `SigLIP`, `SmolLM-base`, `Eval 1 (single-cube task)`, `lerobot framework` (+8 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Blackwell GPU Pin`** (2 nodes): `Blackwell sm_120 (RTX 5090) wheel mismatch`, `torch==2.9.1+cu128 pin`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Proprioception Path`** (2 nodes): `Proprio MLP`, `Proprioception (joint angles)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Text Encoding Path`** (2 nodes): `SmolLM-base`, `Text Encoder`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SmolVLA` connect `SmolVLA Architecture & Performance` to `SmolVLA Finetune Pipeline`, `BC Method Alternatives`, `Robot Setup and Eval Tasks`, `Dataset Schema`, `Vision/Action Encoder Stack`, `Proprioception Path`, `Text Encoding Path`?**
  _High betweenness centrality (0.368) - this node is a cross-community bridge._
- **Why does `101-episode Eval 2 dataset (projet3-eval2-v1-tom-hugo)` connect `Dataset Schema` to `SmolVLA Finetune Pipeline`, `BC Method Alternatives`, `SmolVLA Architecture & Performance`, `Robot Setup and Eval Tasks`?**
  _High betweenness centrality (0.100) - this node is a cross-community bridge._
- **Why does `ACT` connect `BC Method Alternatives` to `Proprioception Path`, `SmolVLA Architecture & Performance`, `Dataset Schema`, `Vision/Action Encoder Stack`?**
  _High betweenness centrality (0.095) - this node is a cross-community bridge._
- **What connects `Rounded box with centered text.`, `SigLIP`, `SmolLM-base` to the rest of the system?**
  _13 weakly-connected nodes found - possible documentation gaps or missing edges._