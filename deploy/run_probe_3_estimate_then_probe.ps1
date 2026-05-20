# 3-trial probe session with estimate-before-grasp (map_only).
# Protocol per trial:
#   1) home photo(s) captured
#   2) MAP estimate shown before grasp
#   3) manual probe grasp FK
#   4) row appended to mapping_samples.csv

Set-Location $PSScriptRoot\..

$session = "deploy\_snaps\probe_1779205245"
if ($args.Count -ge 1) { $session = $args[0] }

Write-Host "[probe] 3 trials | estimate -> probe -> append map rows"
Write-Host "[probe] session: $session"

python -m toolset.perception.probe_camera_vs_fk `
  --map_only --save_frames --color yellow `
  --session_dir $session --n_trials 3 --k 3 --port COM3 --camera_index 0

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n[post] rebuild refined map + both HTMLs"
python -m toolset.perception.probe_map_data --session_dir $session
python -m toolset.perception.probe_square_postprocess --session_dir $session --gallery_only
python -m toolset.perception.probe_map_visual_report --map_session_dir $session

Write-Host "[done]"
