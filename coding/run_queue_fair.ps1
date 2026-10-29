# Waits for the running queue, then runs the same-recipe patient-wise controls.
Set-Location $PSScriptRoot
while (-not (Test-Path E:\Life_Nero\result\QUEUE_DONE.txt)) { Start-Sleep 30 }
python run_experiments.py patient_fair *> E:\Life_Nero\result\log_queue_fair.txt
"done" | Out-File E:\Life_Nero\result\FAIR_DONE.txt