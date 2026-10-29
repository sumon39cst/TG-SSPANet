Set-Location $PSScriptRoot
while (-not (Test-Path E:\Life_Nero\result\ROUND2_DONE.txt)) { Start-Sleep 30 }
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
python 16_round2_eval.py *> E:\Life_Nero\result\log_round2_eval.txt
python 12_revision_stats.py *> E:\Life_Nero\result\log_revision_stats.txt
python 18_round2_figures.py occlusion qualitative *> E:\Life_Nero\result\log_round2_figs.txt
"done" | Out-File E:\Life_Nero\result\ROUND2_POST_DONE.txt