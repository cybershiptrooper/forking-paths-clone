"""Revised-optimizer fits at sentence gap 1 (within-chunk reads always on), in the mask_judge_0925 namespace."""
from pathlib import Path
from ..monitor_tuning_0924 import cohort,worker,optimizer_worker
from .gap1 import Gap1Pool

NEW=Path('results/prompt_bias_v2/mask_judge_0925')
cohort.ROOT=NEW
worker.ROOT=NEW
worker.TuningPool=Gap1Pool  # worker.configure() then installs it as run.CellPool; controls and identity check use it too

if __name__=='__main__':
 optimizer_worker.main()
