"""Revised-optimizer fits (sparse initialisation, fixed size penalty 10) in a fresh namespace.

Reuses the validated tuning worker and optimizer unchanged; only the results root moves, so the
stopped study's STOP sentinels stay in place.
"""
from pathlib import Path
from ..monitor_tuning_0924 import cohort,worker,optimizer_worker

NEW=Path('results/prompt_bias_v2/mask_judge_0925')
cohort.ROOT=NEW
worker.ROOT=NEW

if __name__=='__main__':
 optimizer_worker.main()
