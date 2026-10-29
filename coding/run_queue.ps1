# Detached launcher: runs the whole (resumable) experiment queue, then the XAI evaluations.
Set-Location $PSScriptRoot
python run_experiments.py main ablation patient *> E:\Life_Nero\result\log_queue.txt
python 03_xai_eval.py --group main --tag random_seed0 *> E:\Life_Nero\result\log_xai.txt
python 03_xai_eval.py --group ablation --tag random_seed0 --all_runs *> E:\Life_Nero\result\log_xai_ablation.txt
"done" | Out-File E:\Life_Nero\result\QUEUE_DONE.txt