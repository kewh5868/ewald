# Trigger Folders

Each subdirectory is a deployment package with one `run.sh` entry point. The
defaults are local smoke tests using small fixtures. On Alpine, set the same
environment variables to production paths under `$SCRATCH` or `$PROJECT`.

```text
00_fetch_hybrid3_structures/   HybriD3 API/HTML ingestion into an EWALD catalog
01_generate_simulations/       clean GIWAXS simulation generation, including optional multi-detector sweeps
02_apply_artifacts/            stochastic detector/surface artifact augmentation and labels
03_train_ranker/               lightweight vector-ranker checkpoint build
04_feedback_evaluate/          incremental feedback metrics
05_export_structure_guesses/   copy top-k known structure files per ranked image
```

The training trigger currently builds a baseline vector-ranker checkpoint. It is
not a deep-learning training job; it exists so Alpine deployment, manifests, and
feedback metrics can be tested end-to-end before the heavier model is added.
The guess-export trigger uses that checkpoint to rank artifact images and write
per-sample folders containing copied top-k structure files plus
`ranked_guesses.json`.

For Alpine-scale preparation, use
`data_training/configs/simulation_alpine_fibril_training.example.yaml` as the
clean simulation plan. It sweeps multiple detector geometries and incident
angles before `02_apply_artifacts` expands each clean image across detector and
surface-scattering profiles.
