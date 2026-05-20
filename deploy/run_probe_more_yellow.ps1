# Resume yellow map probing (trials 007+ in existing session).
# Run from an interactive terminal (needs Enter at each cube).
Set-Location $PSScriptRoot\..

$session = "deploy\_snaps\probe_1779205245"
$n = 6
if ($args.Count -ge 1) { $n = [int]$args[0] }

Write-Host "[probe] $n new yellow trials -> $session"
Write-Host "  Each trial: home photo -> MAP estimate -> grasp FK -> est vs meas -> mapping_samples.csv"
python -m toolset.perception.probe_camera_vs_fk `
  --map_only --save_frames --color yellow `
  --session_dir $session --n_trials $n --port COM3 --camera_index 0

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n[post] square fit + refined map + HTML report..."
python -m toolset.perception.probe_square_postprocess --session_dir $session
python -m toolset.perception.probe_map_data --session_dir $session
python -m toolset.perception.probe_map_feasibility --session_dir $session
python -m toolset.perception.probe_map_visual_report --map_session_dir $session
Write-Host "[done] open deploy\_snaps\probe_1779205245\probe_map_explainer.html"
