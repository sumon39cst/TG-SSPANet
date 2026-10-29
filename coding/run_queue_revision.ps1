# Revision experiments (reviewer major comments 3-7), detached and resumable.
Set-Location $PSScriptRoot
python run_experiments.py revision *> E:\Life_Nero\result\log_queue_revision.txt
"done" | Out-File E:\Life_Nero\result\REVISION_DONE.txt