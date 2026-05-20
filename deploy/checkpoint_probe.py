"""
Inspect local or Hugging Face checkpoints and report deploy metadata.

SAC (.pt): state dim, CNN depth, image size, optional keys in the pickle.
ACT (folder / HF repo): LeRobot config — env_state dim, image keys, action dim.

Optional sidecar: <stem>.manifest.json next to the file (see notes/deploy_checkpoints.md).
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_DEPLOY = Path(__file__).resolve().parent
if str(_DEPLOY) not in sys.path:
    sys.path.insert(0, str(_DEPLOY))


@dataclass
class CheckpointInfo:
    path: str
    backend: str  # "sac" | "rlpd" | "act" | "unknown"
    resolved_path: str
    manifest_path: str | None = None
    manifest: dict[str, Any] = field(default_factory=dict)

    # SAC
    n_state: int | None = None
    colour_conditioned: bool | None = None
    use_bowl_xyz: bool | None = None
    n_conv: int | None = None
    image_size: int | None = None  # legacy: square side when H==W; else None
    sac_flatten_dim: int | None = None
    sac_rgb_emb_dim: int | None = None
    sac_policy_h: int | None = None
    sac_policy_w: int | None = None
    global_step: int | str | None = None
    ckpt_keys: list[str] = field(default_factory=list)

    # ACT
    env_state_dim: int | None = None
    image_keys: list[str] = field(default_factory=list)
    action_dim: int | None = None
    policy_type: str | None = None

    # Deploy hints (merged manifest + heuristics)
    suggested_script: str | None = None
    suggested_home_pose: str | None = None
    warnings: list[str] = field(default_factory=list)


def _manifest_path_for(ckpt_path: Path) -> Path | None:
    if ckpt_path.is_dir():
        p = ckpt_path / "deploy_manifest.json"
        return p if p.is_file() else None
    p = ckpt_path.with_name(ckpt_path.stem + ".manifest.json")
    return p if p.is_file() else None


def _load_manifest(ckpt_path: Path) -> tuple[dict[str, Any], str | None]:
    mp = _manifest_path_for(ckpt_path)
    if mp is None:
        return {}, None
    with open(mp, encoding="utf-8") as f:
        return json.load(f), str(mp)


def _detect_sac_n_conv(encoder_state: dict) -> int:
    return 3 if "conv.4.weight" in encoder_state else 2


def probe_sac(path: Path) -> CheckpointInfo:
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"{path}: expected dict checkpoint, got {type(ckpt)}")

    info = CheckpointInfo(
        path=str(path),
        backend="sac",
        resolved_path=str(path.resolve()),
        ckpt_keys=sorted(ckpt.keys()),
    )
    manifest, mp = _load_manifest(path)
    info.manifest = manifest
    info.manifest_path = mp

    if "encoder" not in ckpt or "actor" not in ckpt:
        info.backend = "unknown"
        info.warnings.append("Missing encoder/actor keys — not a Squint SAC handoff?")
        return info

    n_state = int(ckpt["actor"]["proj.state_proj.0.weight"].shape[1])
    info.n_state = n_state
    info.colour_conditioned = n_state in (18, 21)
    info.use_bowl_xyz = n_state == 21
    if n_state not in (12, 18, 21):
        info.warnings.append(f"Unsupported n_state={n_state} (expected 12, 18, or 21)")

    n_conv = _detect_sac_n_conv(ckpt["encoder"])
    info.n_conv = n_conv
    info.global_step = ckpt.get("global_step", ckpt.get("step", "?"))

    # Dims + policy CNN input HxW (same logic as deploy/eval1_v2/infer_sac_legacy.py).
    _ev2 = _DEPLOY / "eval1_v2"
    if str(_ev2) not in sys.path:
        sys.path.insert(0, str(_ev2))
    import infer_sac_legacy as _sac  # noqa: E402

    fd, rbd, sed, _fus, _ns2, _na = _sac._parse_actor_dims(ckpt["actor"])
    info.sac_flatten_dim = fd
    info.sac_rgb_emb_dim = rbd
    enc = _sac.CNNEncoder(n_conv=n_conv).eval()
    enc.load_state_dict(ckpt["encoder"])
    ph, pw = _sac._infer_policy_hw(enc, n_conv, fd, "cpu")
    info.sac_policy_h, info.sac_policy_w = ph, pw
    info.image_size = ph if ph == pw else None  # legacy field

    info.suggested_script = manifest.get(
        "infer_script", "deploy/eval1_v2/infer_sac_legacy.py"
    )
    info.suggested_home_pose = manifest.get("home_pose", "auto")
    if n_state in (18, 21):
        info.warnings.append(
            "Eval2/colour SAC: pass --goal_color 0..5; use --bowl_xyz if n_state=21."
        )
    return info


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def probe_act(path_or_id: str) -> CheckpointInfo:
    local = Path(path_or_id)
    config: dict | None = None
    resolved = path_or_id

    if local.is_dir() and (local / "config.json").is_file():
        config = _read_json(local / "config.json")
        resolved = str(local.resolve())
    else:
        try:
            from lerobot.configs.policies import PreTrainedConfig

            cfg = PreTrainedConfig.from_pretrained(path_or_id)
            config = cfg.to_dict() if hasattr(cfg, "to_dict") else vars(cfg)
            resolved = path_or_id
        except Exception as exc:
            info = CheckpointInfo(
                path=path_or_id,
                backend="unknown",
                resolved_path=resolved,
            )
            info.warnings.append(f"Could not load ACT config: {exc}")
            return info

    info = CheckpointInfo(
        path=path_or_id,
        backend="act",
        resolved_path=resolved,
        policy_type=str(config.get("type", config.get("policy_type", "act"))),
    )
    manifest, mp = _load_manifest(local if local.is_dir() else Path(path_or_id))
    info.manifest = manifest
    info.manifest_path = mp

    features = config.get("input_features") or config.get("features") or {}
    if isinstance(features, dict):
        for key, feat in features.items():
            if not isinstance(feat, dict):
                continue
            shape = feat.get("shape")
            if key == "observation.environment_state" and shape:
                info.env_state_dim = int(shape[0])
            if "images" in key and key.startswith("observation.images."):
                info.image_keys.append(key.replace("observation.images.", ""))
        out = config.get("output_features") or {}
        for key, feat in out.items():
            if key == "action" and isinstance(feat, dict) and feat.get("shape"):
                info.action_dim = int(feat["shape"][0])

    if info.env_state_dim == 10:
        info.suggested_script = manifest.get(
            "infer_script", "deploy/infer_eval1_act.py"
        )
    elif info.env_state_dim == 8:
        info.suggested_script = manifest.get(
            "infer_script", "deploy/infer_eval1_act_nocube.py"
        )
    else:
        info.warnings.append(
            f"env_state_dim={info.env_state_dim!r}: defaulting to 8D nocube script."
        )
        info.suggested_script = manifest.get(
            "infer_script", "deploy/infer_eval1_act_nocube.py"
        )

    info.suggested_home_pose = manifest.get("home_pose", "eval1_rest")
    if not info.image_keys:
        info.image_keys = ["wrist"]
        info.warnings.append("No image keys in config; deploy assumes observation.images.wrist")
    return info


def probe(path_or_id: str) -> CheckpointInfo:
    """Auto-detect SAC .pt vs LeRobot ACT folder / HF repo id."""
    p = Path(path_or_id)
    manifest_backend = None
    if p.is_file() and p.suffix.lower() == ".pt":
        import torch
        raw = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and raw.get("critic") is not None:
            from rlpd_infer_common import probe_rlpd_checkpoint, is_rlpd_checkpoint
            if is_rlpd_checkpoint(raw):
                m = probe_rlpd_checkpoint(p)
                info = CheckpointInfo(
                    path=str(p),
                    backend="rlpd",
                    resolved_path=str(p.resolve()),
                    n_state=m.n_state,
                    colour_conditioned=m.colour_conditioned,
                    use_bowl_xyz=m.use_bowl_xyz,
                    global_step=m.global_step,
                    ckpt_keys=m.ckpt_keys,
                    sac_policy_h=m.image_h,
                    sac_policy_w=m.image_w,
                    manifest=m.manifest,
                    manifest_path=(str(mp) if (mp := _manifest_path_for(p)) else None),
                )
                info.suggested_script = m.manifest.get(
                    "infer_script", "deploy/run_rlpd_ik_place.py"
                )
                info.suggested_home_pose = m.manifest.get("home_pose", "eval1_rest")
                info.warnings.extend(m.warnings)
            else:
                info = probe_sac(p)
        else:
            info = probe_sac(p)
    elif p.is_dir() and (p / "config.json").is_file():
        info = probe_act(str(p))
    elif p.is_file() and p.name == "config.json":
        info = probe_act(str(p.parent))
    else:
        # HF repo id or ambiguous path
        if p.is_file():
            info = CheckpointInfo(
                path=path_or_id,
                backend="unknown",
                resolved_path=str(p.resolve()),
            )
            info.warnings.append("File is not .pt; try a directory or HF repo id for ACT.")
            return info
        try:
            info = probe_act(path_or_id)
        except Exception:
            if p.suffix.lower() == ".pt" and p.is_file():
                info = probe_sac(p)
            else:
                raise

    manifest_backend = (info.manifest or {}).get("backend")
    if manifest_backend in ("sac", "act") and manifest_backend != info.backend:
        info.warnings.append(
            f"manifest backend={manifest_backend!r} differs from detected {info.backend!r}; "
            "manifest wins for routing when using run_checkpoint.py --trust-manifest"
        )
    return info


def format_report(info: CheckpointInfo) -> str:
    lines = [
        f"path:           {info.path}",
        f"resolved:       {info.resolved_path}",
        f"backend:        {info.backend}",
    ]
    if info.manifest_path:
        lines.append(f"manifest:       {info.manifest_path}")
    if info.backend in ("sac", "rlpd"):
        lines += [
            f"n_state:        {info.n_state}  "
            f"(colour={info.colour_conditioned}, bowl_xyz={info.use_bowl_xyz})",
            f"CNN:            n_conv={info.n_conv}, flatten={info.sac_flatten_dim}, "
            f"policy_input_HxW=({info.sac_policy_h}, {info.sac_policy_w}), rgb_emb={info.sac_rgb_emb_dim}",
            f"global_step:    {info.global_step}",
            f"ckpt keys:      {', '.join(info.ckpt_keys[:12])}"
            + (" ..." if len(info.ckpt_keys) > 12 else ""),
        ]
    elif info.backend == "act":
        lines += [
            f"policy_type:    {info.policy_type}",
            f"env_state_dim:  {info.env_state_dim}",
            f"image_keys:     {info.image_keys}",
            f"action_dim:     {info.action_dim}",
        ]
    if info.suggested_script:
        lines.append(
            "suggested_run:  python deploy/run_best_ckpt.py --ckpt <path>   "
            f"(same as run_checkpoint.py; engine: {info.suggested_script})"
        )
    if info.suggested_home_pose:
        lines.append(f"home_pose:      {info.suggested_home_pose}")
    for w in info.warnings:
        lines.append(f"WARNING:        {w}")
    return "\n".join(lines)


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help=".pt file, ACT folder, or HuggingFace repo id")
    ap.add_argument("--json", action="store_true", help="Print JSON instead of text")
    args = ap.parse_args()
    info = probe(args.path)
    if args.json:
        print(json.dumps(asdict(info), indent=2))
    else:
        print(format_report(info))


if __name__ == "__main__":
    main()
