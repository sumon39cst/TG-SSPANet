Set-Location $PSScriptRoot
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
for ($i = 0; $i -lt 6; $i++) {
    python 11_revision_eval.py *> "E:\Life_Nero\result\log_revision_eval_$i.txt"
    if ((Get-Content "E:\Life_Nero\result\log_revision_eval_$i.txt" -Tail 1) -match "^done") { break }
}
python 12_revision_stats.py *> E:\Life_Nero\result\log_revision_stats.txt
python 13_revision_figures.py *> E:\Life_Nero\result\log_revision_figures.txt
"done" | Out-File E:\Life_Nero\result\REVEVAL_DONE.txt