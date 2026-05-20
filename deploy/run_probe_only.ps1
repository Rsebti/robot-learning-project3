# Grasp FK only — compare to estimate from a prior capture folder.
# Default: test_probe_1779225334 (analyze_test_images output).
Set-Location $PSScriptRoot\..

$map = "deploy\_snaps\probe_1779205245"
$from = "deploy\_snaps\test_probe_1779225334"
if ($args.Count -ge 1) { $from = $args[0] }

Write-Host "[probe_only] estimate from: $from"
Write-Host "[probe_only] cube must still be at the same spot.`n"

python deploy/probe_live_estimate_grasp.py `
  --map_session_dir $map `
  --probe_only --from_dir $from
