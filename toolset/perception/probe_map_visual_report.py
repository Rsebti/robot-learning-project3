"""
probe_map_visual_report.py — HTML report explaining what probe numbers mean.

Usage:
    python -m toolset.perception.probe_map_visual_report `
        --map_session_dir deploy/_snaps/probe_1779205245

    python -m toolset.perception.probe_map_visual_report `
        --map_session_dir deploy/_snaps/probe_1779205245 `
        --live_dirs deploy/_snaps/live_probe_1779222592
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from toolset.perception.probe_map_data import (
    load_excluded_trials,
    load_mapping_rows,
    trial_quality_flags,
)


def _esc(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _rel_src(path: Path | None, html_dir: Path) -> str | None:
    if path is None or not path.is_file():
        return None
    path = path.resolve()
    html_dir = html_dir.resolve()
    try:
        return path.relative_to(html_dir).as_posix()
    except ValueError:
        return Path(os.path.relpath(path, html_dir)).as_posix()


def _trial_home_dir(map_dir: Path, trial: int) -> Path:
    return map_dir / f"trial_{trial:03d}" / "home"


def _pick_trial_image(trial_home: Path, *, prefer: str = "final_overlay") -> Path | None:
    """Best available still for a training trial."""
    stem = "frame_00"
    order = {
        "final_overlay": [
            f"{stem}_final_overlay.png",
            f"{stem}_square_overlay.png",
            f"{stem}_overlay.png",
            f"{stem}.png",
        ],
        "raw": [f"{stem}.png", f"{stem}_final_overlay.png"],
    }[prefer]
    for name in order:
        p = trial_home / name
        if p.is_file():
            return p
    for p in sorted(trial_home.glob("frame_*.png")):
        if "_mask" not in p.name and "_overlay" not in p.name:
            return p
    return None


def _figure(path: Path | None, html_dir: Path, caption: str, *, css: str = "") -> str:
    src = _rel_src(path, html_dir)
    if not src:
        return (
            f'<figure class="cell missing {css}">'
            f'<div class="ph">no image</div>'
            f"<figcaption>{_esc(caption)}</figcaption></figure>"
        )
    return (
        f'<figure class="cell {css}">'
        f'<img src="{_esc(src)}" alt="{_esc(caption)}" loading="lazy">'
        f"<figcaption>{caption}</figcaption></figure>"
    )


def _trial_caption(row: dict, dist_px: float | None = None) -> str:
    fk = row["fk"]
    extra = f" · <b>{dist_px:.0f} px</b> from your (du,dv)" if dist_px is not None else ""
    return (
        f"<b>trial {row['trial']:03d}</b>{extra}<br>"
        f"du={row['du']:+.0f} dv={row['dv']:+.0f}<br>"
        f"FK ({fk[0]:+.3f}, {fk[1]:+.3f}, {fk[2]:+.3f}) m"
    )


def _live_image_candidates(live_dir: Path) -> dict[str, Path | None]:
    home = live_dir / "home"
    return {
        "overlay": home / "live_estimate_overlay.png" if home.is_dir() else None,
        "frame_last": home / "frame_last.png" if home.is_dir() else None,
        "frame_00": home / "frame_00.png" if home.is_dir() else None,
    }


def _comparison_gallery(
    live_dir: Path,
    map_dir: Path,
    est: dict,
    trial_map: dict[int, dict],
    html_dir: Path,
) -> str:
    du, dv = float(est["du_px"]), float(est["dv_px"])
    nearest_tri = int(est["nearest_train_trial"])
    neighbor_trials = [int(t) for t in est.get("neighbor_trials", [])]

    live_cands = _live_image_candidates(live_dir)
    live_overlay = live_cands.get("overlay")
    if live_overlay is not None and not live_overlay.is_file():
        live_overlay = None
    live_raw = live_cands.get("frame_last") or live_cands.get("frame_00")
    if live_raw is not None and not live_raw.is_file():
        live_raw = None

    near_row = trial_map.get(nearest_tri)
    near_home = _trial_home_dir(map_dir, nearest_tri)
    near_img = _pick_trial_image(near_home)

    live_cap = (
        f"<b>Your live spot</b> ({_esc(est.get('label', live_dir.name))})<br>"
        f"du={du:+.0f} dv={dv:+.0f} · pixel ({est['pixel_u']:.0f}, {est['pixel_v']:.0f})<br>"
        f"MAP est ({est['est_xyz_user_m'][0]:+.3f}, "
        f"{est['est_xyz_user_m'][1]:+.3f}, {est['est_xyz_user_m'][2]:+.3f}) m"
    )
    near_cap = "Closest training probe (image-space)"
    if near_row:
        near_cap = _trial_caption(near_row, float(est.get("nearest_train_dist_px", 0)))

    primary = f"""
    <h3>Side-by-side: live vs closest training trial</h3>
    <p class="hint">Same wrist home pose style. Yellow dot on live = detected centroid.
    Green verify overlay on training = mask used for that map row.</p>
    <div class="compare-row primary">
      {_figure(live_overlay or live_raw, html_dir, live_cap, css="live")}
      {_figure(near_img, html_dir, near_cap, css="nearest")}
    </div>
    """

    if live_overlay and live_raw and live_raw != live_overlay:
        primary += f"""
    <div class="compare-row">
      {_figure(live_raw, html_dir, "Live raw frame (last capture)", css="live")}
      {_figure(_pick_trial_image(near_home, prefer="raw"), html_dir,
               f"trial {nearest_tri:03d} raw home frame", css="nearest")}
    </div>
    """

    neighbor_figs = []
    dists = est.get("neighbor_dist_px")
    for i, tri in enumerate(neighbor_trials):
        row = trial_map.get(tri)
        if row is None:
            continue
        d_px = float(dists[i]) if dists and i < len(dists) else float(
            np.hypot(row["du"] - du, row["dv"] - dv),
        )
        img = _pick_trial_image(_trial_home_dir(map_dir, tri))
        neighbor_figs.append(
            _figure(img, html_dir, _trial_caption(row, d_px), css="neighbor"),
        )

    neighbors_html = ""
    if neighbor_figs:
        neighbors_html = f"""
    <h3>kNN neighbors blended into MAP estimate</h3>
    <p class="hint">Trials {neighbor_trials} — weighted by inverse image distance at your (du, dv).</p>
    <div class="compare-row neighbors">
      {"".join(neighbor_figs)}
    </div>
    """

    missing_live = ""
    if live_overlay is None and live_raw is None:
        missing_live = (
            '<p class="warn">Live folder has no <code>home/live_estimate_overlay.png</code> '
            "or <code>frame_last.png</code> — re-run "
            "<code>probe_live_estimate_grasp.py</code> to save frames. "
            "Training images below still show what the map borrowed from.</p>"
        )

    return primary + neighbors_html + missing_live


def _live_blocks(
    live_dirs: list[Path],
    map_dir: Path,
    trial_map: dict[int, dict],
    html_dir: Path,
) -> str:
    parts: list[str] = []
    for d in live_dirs:
        d = d.resolve()
        est_path = d / "live_estimate.json"
        grasp_path = d / "grasp" / "grasp_record.json"
        if not est_path.is_file():
            continue
        est = json.loads(est_path.read_text(encoding="utf-8"))
        meas = None
        if grasp_path.is_file():
            meas = json.loads(grasp_path.read_text(encoding="utf-8")).get("fk_xyz_user_m")

        ex, ey, ez = est["est_xyz_user_m"]
        nx, ny, nz = est["nearest_train_xyz_m"]
        gallery = _comparison_gallery(d, map_dir, est, trial_map, html_dir)

        meas_row = ""
        if meas:
            mx, my, mz = meas
            err_y = (ey - my) * 1000
            err_xy = float(np.linalg.norm([ex - mx, ey - my]) * 1000)
            meas_row = f"""
            <tr class="meas"><td><strong>Your grasp FK</strong> (torque off, aligned, Enter)</td>
                <td>{mx:+.3f}</td><td>{my:+.3f}</td><td>{mz:+.3f}</td>
                <td>Ground truth for <em>this</em> cube spot</td></tr>
            <tr class="warn"><td>Estimate − measured</td>
                <td colspan="3">Δxy {err_xy:.0f} mm · Δy {err_y:+.0f} mm</td>
                <td>{'OK for IK try' if err_xy < 40 else 'Too far — fix map or re-probe here'}</td></tr>
            """
        else:
            meas_row = (
                '<tr><td colspan="5"><em>No grasp recorded in this folder '
                "(Enter probe not done?)</em></td></tr>"
            )

        parts.append(f"""
        <section class="card live">
          <h2>Live test: {_esc(d.name)}</h2>
          <p class="tag">{_esc(est.get('label', ''))}</p>
          {gallery}
          <table>
            <tr><th>What</th><th>X (right)</th><th>Y (forward)</th><th>Z (up)</th><th>Meaning</th></tr>
            <tr><td>Home pixel (u, v)</td><td colspan="3">{est['pixel_u']:.0f}, {est['pixel_v']:.0f}</td>
                <td>Where cube looked in wrist image</td></tr>
            <tr><td>Offset du, dv (px)</td><td colspan="3">{est['du_px']:+.0f}, {est['dv_px']:+.0f}</td>
                <td>From image center (320, 240)</td></tr>
            <tr class="est"><td><strong>MAP estimate</strong> (kNN, no robot move yet)</td>
                <td>{ex:+.3f}</td><td>{ey:+.3f}</td><td>{ez:+.3f}</td>
                <td>IK / approach would use this XY,Z</td></tr>
            <tr><td>Closest old trial #{est['nearest_train_trial']}</td>
                <td>{nx:+.3f}</td><td>{ny:+.3f}</td><td>{nz:+.3f}</td>
                <td>{est['nearest_train_dist_px']:.0f} px away in image</td></tr>
            <tr><td>kNN neighbors</td><td colspan="4">trials {est['neighbor_trials']}</td></tr>
            {meas_row}
          </table>
        </section>
        """)
    return "\n".join(parts) if parts else "<p><em>No live_probe_* folders passed.</em></p>"


def _gallery_block(
    title: str,
    hint: str,
    rows: list[dict],
    map_dir: Path,
    html_dir: Path,
    *,
    css: str = "train",
    badge: str = "",
) -> str:
    if not rows:
        return ""
    cells = []
    for r in sorted(rows, key=lambda x: x["trial"]):
        img = _pick_trial_image(_trial_home_dir(map_dir, int(r["trial"])))
        cap = _trial_caption(r)
        if badge:
            cap = f"<span class='badge {_esc(css)}'>{_esc(badge)}</span><br>" + cap
        flags = trial_quality_flags(r)
        if flags:
            cap += "<br><span class='flags'>" + "; ".join(_esc(f) for f in flags) + "</span>"
        cells.append(_figure(img, html_dir, cap, css=css))
    return f"""
  <div class="card">
    <h2>{_esc(title)}</h2>
    <p class="hint">{hint}</p>
    <div class="compare-row {css}">
      {"".join(cells)}
    </div>
  </div>
  """


def _training_galleries(
    map_dir: Path,
    active: list[dict],
    all_rows: list[dict],
    excluded: set[int],
    html_dir: Path,
) -> str:
    excl_rows = [r for r in all_rows if int(r["trial"]) in excluded]
    review_rows = [
        r for r in all_rows
        if int(r["trial"]) not in excluded and trial_quality_flags(r)
    ]
    parts = [
        _gallery_block(
            "Active map probes",
            "Used in kNN — check photos for hand in frame.",
            active, map_dir, html_dir, css="train",
        ),
    ]
    if excl_rows:
        parts.append(_gallery_block(
            "Excluded from map",
            "excluded_trials.yaml",
            excl_rows, map_dir, html_dir, css="excluded", badge="EXCLUDED",
        ))
    if review_rows:
        parts.append(_gallery_block(
            "Review — suspicious FK (still in map)",
            "Add to excluded: probe_map_data --session_dir ... --exclude N",
            review_rows, map_dir, html_dir, css="review", badge="REVIEW",
        ))
    return "\n".join(parts)


def build_html(
    map_dir: Path,
    rows: list[dict],
    all_rows: list[dict],
    excluded: set[int],
    live_dirs: list[Path],
    html_dir: Path,
) -> str:
    map_dir = map_dir.resolve()
    trial_map = {int(r["trial"]): r for r in rows}
    color = "?"
    sj = map_dir / "session.json"
    if sj.is_file():
        color = json.loads(sj.read_text(encoding="utf-8")).get("color", "?")

    train_rows = ""
    for r in rows:
        fk = r["fk"]
        tri = int(r["trial"])
        thumb = _pick_trial_image(_trial_home_dir(map_dir, tri))
        thumb_html = ""
        src = _rel_src(thumb, html_dir)
        if src:
            thumb_html = (
                f'<img class="thumb" src="{_esc(src)}" alt="trial {tri:03d}" '
                f'title="trial {tri:03d} home">'
            )
        train_rows += f"""
        <tr>
          <td class="thumb-cell">{thumb_html}<strong>trial {tri:03d}</strong></td>
          <td>{r['du']:+.0f}</td><td>{r['dv']:+.0f}</td>
          <td>{r['pixel_u']:.0f}</td><td>{r['pixel_v']:.0f}</td>
          <td class="fk">{fk[0]:+.3f}</td><td class="fk">{fk[1]:+.3f}</td><td class="fk">{fk[2]:+.3f}</td>
          <td>FK at grasp when you built the map</td>
        </tr>
        """

    ys = [r["fk"][1] for r in rows]
    xs = [r["fk"][0] for r in rows]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Probe map — what the numbers mean</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 24px auto; padding: 0 16px;
            background: #1a1a1e; color: #e8e8ec; line-height: 1.5; }}
    h1 {{ font-size: 1.4rem; }}
    h2 {{ font-size: 1.15rem; margin-top: 2rem; color: #9cf; }}
    h3 {{ font-size: 1rem; color: #bdf; margin: 1.2rem 0 0.5rem; }}
    .card {{ background: #252530; border-radius: 10px; padding: 16px 20px; margin: 16px 0;
             border: 1px solid #3a3a48; }}
    .legend {{ background: #2a2a38; padding: 12px 16px; border-left: 4px solid #6af; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.92rem; }}
    th, td {{ border: 1px solid #444; padding: 8px 10px; text-align: left; vertical-align: middle; }}
    th {{ background: #333; }}
    tr.est td {{ background: #2a3a2a; }}
    tr.meas td {{ background: #2a2a4a; }}
    tr.warn td {{ background: #4a3a2a; color: #fc8; }}
    tr.fk td {{ color: #8f8; }}
    .thumb-cell {{ min-width: 100px; }}
    img.thumb {{ width: 72px; height: auto; display: block; margin-bottom: 4px;
                 border: 1px solid #555; border-radius: 4px; }}
    .axis {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin: 16px 0; }}
    .axis div {{ background: #2d2d3a; padding: 12px; border-radius: 8px; }}
    .flow {{ font-family: ui-monospace, monospace; font-size: 0.85rem; white-space: pre-wrap;
             background: #111; padding: 12px; border-radius: 8px; }}
    .tag {{ color: #8cf; }}
    .hint {{ color: #aaa; font-size: 0.88rem; margin: 0.4rem 0 0.8rem; }}
    p.warn {{ color: #fc8; font-size: 0.9rem; }}
    .compare-row {{ display: flex; flex-wrap: wrap; gap: 14px; margin: 12px 0 20px;
                    align-items: flex-start; }}
    .compare-row.primary .cell.live {{ border-color: #6cf; }}
    .compare-row.primary .cell.nearest {{ border-color: #fc6; }}
    .cell {{ flex: 1 1 280px; max-width: 340px; text-align: center;
             background: #1e1e28; border-radius: 8px; padding: 8px;
             border: 2px solid #444; }}
    .cell img {{ max-width: 100%; height: auto; display: block; margin: 0 auto;
                 border: 1px solid #333; background: #000; }}
    .cell.missing .ph {{ height: 180px; display: flex; align-items: center;
                         justify-content: center; color: #666; background: #111;
                         border: 1px dashed #444; border-radius: 4px; }}
    .cell figcaption {{ font-size: 0.78rem; color: #bbb; margin-top: 8px; line-height: 1.35; }}
    .cell.excluded {{ border-color: #a44; opacity: 0.85; }}
    .cell.review {{ border-color: #c84; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.72rem;
              font-weight: bold; margin-bottom: 4px; }}
    .badge.excluded {{ background: #522; color: #faa; }}
    .badge.review {{ background: #532; color: #fc8; }}
    .flags {{ color: #f96; font-size: 0.72rem; }}
    tr.excluded td {{ opacity: 0.55; text-decoration: line-through; }}
    a {{ color: #9df; }}
  </style>
</head>
<body>
  <h1>Probe map — what each number represents</h1>
  <p>Session: <code>{_esc(str(map_dir))}</code> · color: <strong>{_esc(color)}</strong></p>

  <div class="card legend">
    <h2>User frame (all xyz below)</h2>
    <div class="axis">
      <div><strong>X</strong> — right +</div>
      <div><strong>Y</strong> — forward + (toward back of table)</div>
      <div><strong>Z</strong> — up + (height)</div>
    </div>
    <p>Z for cube center is usually ~table + half cube (~1–2 cm). Y is often where live map looks wrong.</p>
  </div>

  {_training_galleries(map_dir, rows, all_rows, excluded, html_dir)}

  <div class="card">
    <h2>Pipeline A — Building the map (your probing sessions)</h2>
    <div class="flow">Cube still → HOME photo → pixel (du,dv)
    → you grasp SAME cube → FK xyz stored
    → row in mapping_samples.csv

    Y in CSV = FK at grasp, NOT guessed from other trials.</div>
    <p><strong>Not the same as live test:</strong> probing never interpolated Y from other trials.</p>
    <table>
      <tr>
        <th>Row</th><th>du</th><th>dv</th><th>u</th><th>v</th>
        <th>X</th><th>Y</th><th>Z</th><th>Represents</th>
      </tr>
      {train_rows}
    </table>
    <p>Training span: X [{min(xs):+.2f}, {max(xs):+.2f}] m ·
       Y [{min(ys):+.2f}, {max(ys):+.2f}] m</p>
  </div>

  <div class="card">
    <h2>Pipeline B — Live test + IK (new cube position)</h2>
    <div class="flow">HOME photo only → refined mask → (du,dv)
    → kNN blend of OLD trials → MAP estimate xyz
    → (optional) you grasp → measured FK
    → IK uses ESTIMATE unless you override

    If Y is "super off" here, estimate ≠ where cube really is.</div>
  </div>

  {_live_blocks(live_dirs, map_dir, trial_map, html_dir)}

  <div class="card">
    <h2>What to trust for motion</h2>
    <table>
      <tr><th>Use case</th><th>Trust</th><th>Do not trust</th></tr>
      <tr><td>Pick training region</td><td>FK rows in table above</td><td>—</td></tr>
      <tr><td>New spot, no grasp yet</td><td>Rough XY from map</td><td>Y if far outside training band</td></tr>
      <tr><td>Before IK</td><td>Measured grasp FK after Enter-probe</td><td>Map estimate if Δy &gt; ~4 cm vs measured</td></tr>
      <tr><td>run approach / ik_relative</td><td>Measured xy or good estimate</td><td>Bad map Y → arm goes too far forward</td></tr>
    </table>
  </div>

  <p style="color:#888;font-size:0.85rem">Generated by probe_map_visual_report.py</p>
</body>
</html>"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map_session_dir", type=Path, required=True)
    p.add_argument("--live_dirs", type=Path, nargs="*", default=[],
                   help="live_probe_* folders (repeat flag or pass parent to glob)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    map_dir = args.map_session_dir.resolve()
    excluded = load_excluded_trials(map_dir)
    all_rows = load_mapping_rows(map_dir, use_refined=True, include_excluded=True)
    rows = load_mapping_rows(map_dir, use_refined=True)
    if not rows:
        raise SystemExit(f"No active map rows in {map_dir} (excluded: {sorted(excluded)})")

    live_dirs: list[Path] = list(args.live_dirs)
    if not live_dirs and (PROJECT_ROOT / "deploy" / "_snaps").is_dir():
        live_dirs = sorted((PROJECT_ROOT / "deploy" / "_snaps").glob("live_probe_*"))

    out = args.out or (map_dir / "probe_map_explainer.html")
    out = out.resolve()
    html = build_html(map_dir, rows, all_rows, excluded, live_dirs, out.parent)
    out.write_text(html, encoding="utf-8")
    print(f"[report] wrote {out}")
    print(f"[report] open in browser: file:///{out.as_posix()}")


if __name__ == "__main__":
    main()
