# Archived experiment scripts (deprecated)

These one-off PowerShell wrappers were replaced by the unified runner:

- `scripts/run_experiment.ps1`
- `scripts/run_experiment.py`
- `experiments/aviation.yaml`

## Migration map

| Old script | New command |
|------------|-------------|
| `run_p0_ablations.ps1` | `.\scripts\run_experiment.ps1 -Grid p0_ablations` |
| `run_exp04_fastlr_grid.ps1` | `.\scripts\run_experiment.ps1 -Grid exp04_fastlr` |
| `run_exp04_full.ps1` | `.\scripts\run_experiment.ps1 -Exp exp04_full` |
| `run_final_eval_exp04.ps1` | `.\scripts\run_experiment.ps1 -Exp exp04_eval` |
| `run_eval_exp04_as_protonet.ps1` | `.\scripts\run_experiment.ps1 -Exp exp04_eval_as_protonet` |
| `run_exp05_full.ps1` | `.\scripts\run_experiment.ps1 -Exp exp05_full` |
| `run_exp06_full.ps1` | `.\scripts\run_experiment.ps1 -Exp exp06_full` |
| `run_final_eval_exp06.ps1` | `.\scripts\run_experiment.ps1 -Exp exp06_eval` |
| `run_exp07_smoke.ps1` | `.\scripts\run_experiment.ps1 -Exp exp07_smoke` |
| `run_exp07_adapt_grid.ps1` | `.\scripts\run_experiment.ps1 -Grid exp07_adapt` |
| `run_exp07_full.ps1` | `.\scripts\run_experiment.ps1 -Exp exp07_full` |
| `run_final_eval_exp07.ps1` | `.\scripts\run_experiment.ps1 -Exp exp07_eval` |
| `run_exp08_protonet_smoke.ps1` | `.\scripts\run_experiment.ps1 -Exp exp08_protonet_smoke` |
| `run_exp08_protonet_full.ps1` | `.\scripts\run_experiment.ps1 -Exp exp08_protonet` |
| `run_final_eval_exp08.ps1` | `.\scripts\run_experiment.ps1 -Exp exp08_eval` |
| `run_final_eval_exp08_exclude22.ps1` | `.\scripts\run_experiment.ps1 -Exp exp08_eval_exclude22` |
| `run_final_eval_exp08_qc_deleted.ps1` | `.\scripts\run_experiment.ps1 -Exp exp08_eval_qc_deleted` |
| `run_diagnose_hard_tasks.ps1` | `.\scripts\run_experiment.ps1 -Exp exp08_diagnose` |

Utility scripts kept as archive reference:

- `exclude_specimen22_damage.ps1` — logic moved into `exp08_eval_exclude22` via YAML `exclude_sample_ids_from`
- `list_domain3_damage_qc.ps1` — manual QC listing only

Python equivalent (cross-platform):

```bash
python scripts/run_experiment.py --list
python scripts/run_experiment.py --exp exp08_protonet --stage train
python scripts/run_experiment.py --exp exp08_eval
python scripts/run_experiment.py --grid exp04_fastlr
```

Override any arg: `--set iters=20 --set fast_lr=0.01`

Dry run: `--dry-run`
