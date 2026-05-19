"""
import_checkerboard_intrinsics.py — write camera_intrinsics.npz for CubeLocalizer.

Writes intrinsics at native calibration resolution by default (1920x1080).
Pass --deploy_width / --deploy_height only when the OpenCV stream differs.

Usage:
  python -m toolset.calibration_scripts.import_checkerboard_intrinsics
  python -m toolset.calibration_scripts.import_checkerboard_intrinsics \\
      --deploy_width 640 --deploy_height 480
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "toolset" / "configs" / "kinematics"

# Friend's checkerboard solve @ 1920x1080 (reproj ~3.05 px)
FRIEND_NATIVE = {
    "image_size": [1920, 1080],
    "mean_reproj_err_px": 3.0503,
    "fx": 678.91,
    "fy": 677.98,
    "cx": 963.03,
    "cy": 563.69,
    "dist": [
        0.03445741546881645,
        -0.05711470904549814,
        0.0005838923219725458,
        0.00038779296341359427,
        0.011379063813159436,
    ],
}


def intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def scale_intrinsics(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    dist: np.ndarray,
    size_native: tuple[int, int],
    size_deploy: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    w0, h0 = size_native
    w1, h1 = size_deploy
    sx, sy = w1 / w0, h1 / h0
    K = intrinsics_matrix(fx * sx, fy * sy, cx * sx, cy * sy)
    return K, dist.astype(float).reshape(-1), [w1, h1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--deploy_width",
        type=int,
        default=None,
        help="Deploy stream width (default: native calibration width)",
    )
    p.add_argument(
        "--deploy_height",
        type=int,
        default=None,
        help="Deploy stream height (default: native calibration height)",
    )
    p.add_argument("--out_dir", type=Path, default=OUT_DIR)
    args = p.parse_args()

    native = tuple(FRIEND_NATIVE["image_size"])
    if args.deploy_width is None and args.deploy_height is None:
        deploy = native
    elif args.deploy_width is None or args.deploy_height is None:
        p.error("pass both --deploy_width and --deploy_height, or neither")
    else:
        deploy = (args.deploy_width, args.deploy_height)

    dist = np.array(FRIEND_NATIVE["dist"], dtype=float)
    if deploy == native:
        K = intrinsics_matrix(
            FRIEND_NATIVE["fx"],
            FRIEND_NATIVE["fy"],
            FRIEND_NATIVE["cx"],
            FRIEND_NATIVE["cy"],
        )
        dist_flat = dist.reshape(-1)
        deploy_size = list(native)
        scale_note = "native (no scale)"
    else:
        K, dist_flat, deploy_size = scale_intrinsics(
            FRIEND_NATIVE["fx"],
            FRIEND_NATIVE["fy"],
            FRIEND_NATIVE["cx"],
            FRIEND_NATIVE["cy"],
            dist,
            native,
            deploy,
        )
        scale_note = f"scaled {native} -> {deploy_size}"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.out_dir / "camera_intrinsics.npz"
    yaml_path = args.out_dir / "camera_intrinsics.yaml"

    np.savez(
        npz_path,
        K=K,
        dist=dist_flat,
        image_size=np.array(deploy_size),
        mean_reproj_err=float(FRIEND_NATIVE["mean_reproj_err_px"]),
        native_image_size=np.array(native),
        source="friend_checkerboard_1920x1080",
    )

    meta = {
        "source": "friend_checkerboard",
        "native_image_size": list(native),
        "deploy_image_size": deploy_size,
        "mean_reproj_err_px_native": FRIEND_NATIVE["mean_reproj_err_px"],
        "native": {k: v for k, v in FRIEND_NATIVE.items() if k != "dist"},
        "native_dist": FRIEND_NATIVE["dist"],
        "deploy": {
            "K": K.tolist(),
            "dist_coeffs": dist_flat.tolist(),
        },
        "note": (
            "T_cam_in_wrist still from legacy camera_calibration.yaml until "
            "calibrate_hand_eye.py is run on this camera mount."
        ),
    }
    with open(yaml_path, "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    print(f"[import-intrinsics] wrote {npz_path}")
    print(f"[import-intrinsics] deploy K:\n{K}")
    print(f"[import-intrinsics] deploy dist: {dist_flat}")
    print(f"[import-intrinsics] {scale_note}")
    print(
        f"[import-intrinsics] native reproj {FRIEND_NATIVE['mean_reproj_err_px']:.3f} px "
        f"(retune hand-eye if XYZ is off)"
    )


if __name__ == "__main__":
    main()
