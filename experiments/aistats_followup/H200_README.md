# H200 experiment handoff

This is a new, isolated GPU replication. Do not copy the old CPU score files into it.
The archive contains the two public raw datasets and only the required source files;
no repository credentials, Git history, manuscript author details, or private data.

On a Linux H200 machine, upload `aistats_h200_bundle.zip` to the current directory.
Use Python 3.11 or newer (3.12 recommended):

```bash
unzip aistats_h200_bundle.zip
cd aistats_h200_bundle
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install torch-geometric==2.8.0 pygod==1.1.0 numpy scipy scikit-learn networkx pandas
nvidia-smi
bash paper/aistats_followup/run_h200.sh
```

The pinned CUDA wheel follows https://pytorch.org/get-started/previous-versions/ .
The installed driver must support that CUDA runtime; the smoke test checks actual
GPU execution before loading the large graphs. No torch-sparse/torch-scatter extension
is required. Use a persistent terminal (e.g., an existing tmux session) for the run.

Runs: Amazon and Tolokers, ten seeds each, 100-epoch full-batch DOMINANT and GAE;
200-tree attribute-only Isolation Forest; fixed positive-degree control. Each score
cache is evaluated with five test splits and 200 calibration draws per comparison.
The primary comparisons have no trimming, matched calibration sizes and test nodes,
and exclude Amazon's 3,305 unlabeled nodes from calibration/testing. Prespecified
partial-label and trimming sensitivities are also run. GPU results are not silently
pooled with CPU results. Graph loading and calibration analysis still use the CPU.

GPU GAE expansion was specified after CPU DOMINANT results began to arrive; it is
additional replication, not part of the original prespecified detector list.
No training hyperparameters or score signs are selected by observed FDP.

Re-run the last command after interruption: completed score/evaluation files are
reused, and incomplete evaluations restart. Environment or source changes cause
an explicit error instead of mixing runs. Results are archived after each complete
seed and on exceptions. The final `gpu_completion.json` is written only after all
60 model/seed evaluations finish. Earlier `gpu_last_failure.json`, if present,
is retained as history and does not override a later successful completion.

Download `h200_results.zip` from this directory and provide it back in this task.
If a run fails, also provide the last terminal error; partial results remain useful.
Expected runtime is not yet benchmarked on your H200. Single-GPU jobs run sequentially
to avoid competing full-batch graph allocations.
