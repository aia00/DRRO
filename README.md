# DRRO

Code and experiment assets for DRRO-style preference optimization and proxy reward modeling workflows.

## Layout

- `mycode/drro_train/`: DRRO training/evaluation pipeline and helper scripts.
- `mycode/proxy_rm/`: proxy reward model data building and training scripts.
- `mycode/wandb/` and `mycode/drro_train/wandb/`: local experiment run artifacts.
- `verl/`: local `verl` codebase used by DRRO experiments.
- `drro_paths.py`: shared path configuration entry point.

## Quick start

```bash
cd mycode
pip install -r requirements.txt
```

Then use scripts under `mycode/drro_train/` or `mycode/proxy_rm/` to run training/evaluation.

## Notes

- This repository was split out from a larger local research workspace.
- Existing run artifacts are kept to preserve experiment traceability.
