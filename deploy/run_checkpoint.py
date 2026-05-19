"""
Unified checkpoint launcher for project3 deploy.

Resolves a local .pt, a LeRobot/HF ACT folder, or a short name under ~/ or
deploy/eval1_v2/, probes metadata, then runs the right infer script.

Examples:
    python deploy/run_checkpoint.py --inspect ~/eval1_ckpt.pt
    python deploy/run_checkpoint.py --name ckpt_best_1 --goal_color 3
    python deploy/run_checkpoint.py --ckpt deploy/eval1_v2/ckpt.pt
    python deploy/run_checkpoint.py --ckpt hudela390/projet3-act-eval1-v1-no-cube \\
        --target_color yellow --bowl_x 0.16 --bowl_y 0.32

For friend's ``*.pt`` (``ckpt.pt``, ``ckpt_best_1``, ``e1100lat``, …): you run
this script or ``run_best_ckpt.py`` — they exec ``eval1_v2/infer_sac_legacy.py``
under the hood (same stack as ``run_eval1_ckpt`` / ``run_hugod_ckpt`` /
``run_ckpt_best1``; those thin wrappers still work).
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

_DEPLOY = Path(__file__).resolve().parent
_PROJECT = _DEPLOY.parent
_EVAL1_V2 = _DEPLOY / "eval1_v2"

if str(_DEPLOY) not in sys.path:
    sys.path.insert(0, str(_DEPLOY))

from checkpoint_probe import CheckpointInfo, probe  # noqa: E402

# Short names -> search paths (first hit wins)
SEARCH_DIRS = [
    Path.home(),
    Path.home() / "checkpoints",
    _EVAL1_V2,
    _DEPLOY / "eval1",
    _PROJECT,
]

DEFAULT_SAC_CKPT = _EVAL1_V2 / "ckpt.pt"

SAC_GOAL_COLORS = {
    0: "red", 1: "blue", 2: "green", 3: "yellow", 4: "purple", 5: "orange",
}
ACT_GOAL_DEFAULT = "yellow"
SAC_GOAL_DEFAULT = 3


def resolve_checkpoint(ckpt: str | None, name: str | None) -> Path | str:
    if ckpt:
        p = Path(ckpt).expanduser()
        if p.exists():
            return p.resolve() if p.is_file() or p.is_dir() else p
        # HuggingFace repo id (no local path)
        if "/" in ckpt and not p.is_absolute():
            return ckpt
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    if not name:
        if DEFAULT_SAC_CKPT.is_file():
            return DEFAULT_SAC_CKPT
        raise FileNotFoundError(
            "Pass --ckpt PATH or --name STEM (e.g. eval1_ckpt -> ~/eval1_ckpt.pt)"
        )

    stem = name if name.endswith(".pt") else name + ".pt"
    for d in SEARCH_DIRS:
        cand = d / stem
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(
        f"No file named {stem!r} in: " + ", ".join(str(d) for d in SEARCH_DIRS)
    )


def _has(flags: list[str], *names: str) -> bool:
    return any(n in flags for n in names)


def _inject_sac_defaults(extra: list[str], info: CheckpointInfo, manifest: dict) -> list[str]:
    out = list(extra)
    m = manifest or info.manifest or {}

    if not _has(out, "--bowl_xyz") and m.get("bowl_xyz"):
        bx, by, bz = m["bowl_xyz"]
        out += ["--bowl_xyz", str(bx), str(by), str(bz)]
    elif not _has(out, "--bowl_xyz"):
        out += ["--bowl_xyz", "0.16", "0.32", "0.00"]

    if not _has(out, "--goal_color"):
        gc = m.get("goal_color", SAC_GOAL_DEFAULT)
        out += ["--goal_color", str(gc)]

    if not _has(out, "--viz", "--no-viz"):
        out += ["--no-viz"]

    if not _has(out, "--n_episodes"):
        out += ["--n_episodes", "1"]

    if not _has(out, "--home", "--no-home") and m.get("home") is False:
        out += ["--no-home"]
    elif not _has(out, "--home_pose") and m.get("home_pose"):
        out += ["--home_pose", str(m["home_pose"])]

    return out


def _inject_act_defaults(extra: list[str], info: CheckpointInfo, manifest: dict) -> list[str]:
    out = list(extra)
    m = manifest or info.manifest or {}

    if not _has(out, "--target_color"):
        tc = m.get("target_color", ACT_GOAL_DEFAULT)
        out += ["--target_color", str(tc)]

    if not _has(out, "--bowl_x"):
        out += ["--bowl_x", str(m.get("bowl_x", 0.16))]
    if not _has(out, "--bowl_y"):
        out += ["--bowl_y", str(m.get("bowl_y", 0.32))]

    if not _has(out, "--follower_port"):
        out += ["--follower_port", str(m.get("follower_port", "COM3"))]
    if not _has(out, "--camera_index"):
        out += ["--camera_index", str(m.get("camera_index", 1))]

    if not _has(out, "--num_episodes"):
        out += ["--num_episodes", "1"]

    # Same as run_act_8d: start from current pose unless user passes --home
    if not _has(out, "--home", "--no-home") and m.get("home") is not True:
        out += ["--no-home"]

    return out


def route(info: CheckpointInfo, trust_manifest: bool) -> Path:
    m = info.manifest or {}
    if trust_manifest and m.get("infer_script"):
        script = _PROJECT / m["infer_script"]
        if script.is_file():
            return script
        script = _DEPLOY / m["infer_script"]
        if script.is_file():
            return script

    if info.backend == "sac":
        return _EVAL1_V2 / "infer_sac_legacy.py"
    if info.backend == "act":
        if info.env_state_dim == 10:
            return _DEPLOY / "infer_eval1_act.py"
        return _DEPLOY / "infer_eval1_act_nocube.py"
    raise RuntimeError(
        f"Cannot route backend={info.backend!r}. Use --backend sac|act or fix the file."
    )


def build_argv(
    resolved: Path | str,
    info: CheckpointInfo,
    extra: list[str],
    *,
    trust_manifest: bool,
) -> list[str]:
    m = info.manifest or {}
    backend = m.get("backend") if trust_manifest and m.get("backend") else info.backend

    if backend == "sac":
        path = Path(resolved)
        extra = _inject_sac_defaults(extra, info, m)
        return ["--checkpoint", str(path)] + extra

    if backend == "act":
        extra = _inject_act_defaults(extra, info, m)
        return ["--policy_path", str(resolved)] + extra

    raise RuntimeError(f"Unknown backend {backend!r}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--ckpt", help="Path to .pt, ACT folder, or HuggingFace repo id")
    ap.add_argument(
        "--name",
        help="Short name (e.g. eval1_ckpt, ckpt_best_1) -> ~/STEM.pt or deploy search dirs",
    )
    ap.add_argument("--inspect", action="store_true", help="Print metadata and exit")
    ap.add_argument("--backend", choices=["auto", "sac", "act"], default="auto")
    ap.add_argument(
        "--trust-manifest",
        action="store_true",
        help="Prefer deploy_manifest.json / stem.manifest.json for backend and script",
    )
    args, extra = ap.parse_known_args()

    resolved = resolve_checkpoint(args.ckpt, args.name)
    info = probe(str(resolved))

    if args.backend != "auto":
        info.backend = args.backend

    if args.inspect:
        from checkpoint_probe import format_report

        print(format_report(info))
        return

    if info.backend not in ("sac", "act"):
        print(f"[run] probe failed:\n", file=sys.stderr)
        from checkpoint_probe import format_report

        print(format_report(info), file=sys.stderr)
        sys.exit(1)

    script = route(info, args.trust_manifest)
    if not script.is_file():
        print(f"[run] missing infer script: {script}", file=sys.stderr)
        sys.exit(1)

    argv_tail = build_argv(resolved, info, extra, trust_manifest=args.trust_manifest)
    sys.argv = [str(script)] + argv_tail

    print(f"[run] backend={info.backend}  script={script.relative_to(_PROJECT)}")
    print(f"[run] checkpoint={resolved}")
    if info.n_conv is not None:
        print(f"[run] SAC CNN n_conv={info.n_conv} image_size={info.image_size} n_state={info.n_state}")
    if info.env_state_dim is not None:
        print(f"[run] ACT env_state_dim={info.env_state_dim} images={info.image_keys}")

    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
