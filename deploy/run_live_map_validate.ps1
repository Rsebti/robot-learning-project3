# Live: home capture -> refined detect -> map estimate vs training -> optional grasp FK probe.
# Run in an interactive terminal at the robot.
Set-Location $PSScriptRoot\..

$map = "deploy\_snaps\probe_1779205245"
$label = "live_validate"
if ($args.Count -ge 1) { $label = $args[0] }

Write-Host "[1/2] Live map estimate (capture + compare to training map)..."
python deploy/probe_live_estimate_grasp.py `
  --map_session_dir $map `
  --label $label `
  --n_frames 10 --port COM3 --camera_index 0

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n[2/2] Refresh HTML report..."
python -m toolset.perception.probe_map_visual_report --map_session_dir $map

Write-Host "`nDone. Check newest live_probe_* folder and probe_map_explainer.html"
