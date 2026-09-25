# NCSA H200 execution

Started through the user's authenticated JupyterLab terminal on 14 September 2026.
Browser: https://jupyter.ncsa.illinois.edu/user/suhaank2/lab

Remote directory: `/home/suhaank2/conformal-graph-anomaly/aistats_h200_bundle`.
The parent checkout was updated with `git pull --ff-only`; existing untracked logs
were preserved. The follow-up code was absent from that checkout, so the explicit
allowlisted bundle was uploaded to the research folder and extracted separately.

Transferred archive SHA-256:
`b0416d9855af9f27169706530f91bdd544b645ee78e6a0433c45861282214a04`.
The remote copy matched before extraction.

Observed hardware: NVIDIA H200 NVL, 143771 MiB total memory. Default Python is
3.13.14 and did not have PyTorch. Setup creates an isolated `.venv` with CUDA
PyTorch 2.13.0, PyG 2.8.0, and PyGOD 1.1.0. Actual installed versions and runtime
properties are recorded by the experiment runner before training.

The owned launcher is `launch_remote.sh`, launched with nohup. Its PID is stored
in `remote_job.pid` (initial PID 318). Setup output is `setup.log`; stage and exit
information is in `remote_status.txt`. Experiment output is
`paper/aistats_followup/h200_run.log`. Results are in `h200_results.zip`.

The launcher progresses from SETTING_UP to RUNNING to COMPLETE, or records FAILED.
Do not infer training has started from SETTING_UP, or completion from a partial
results archive. Confirm `gpu_completion.json` before reporting all 60 model/seed
evaluations as finished. A terminal disconnect should leave the detached job
running; termination of the Jupyter server allocation will not.

This record is operational provenance, not part of the anonymous submission bundle.

## Launch verification

The remote status changed to RUNNING. Both `SMOKE PASSED dominant_pygod cuda`
and `SMOKE PASSED gae cuda` appeared in the experiment log, followed by
`START amazon dominant_pygod 0`. This verifies actual CUDA execution and start
of the full batch. It does not establish completion of any full training run.
Installed dependencies visible at launch include NumPy 2.5.3, SciPy 1.18.1,
scikit-learn 1.9.1, PyG 2.8.0, and PyGOD 1.1.0. Preserve the complete
`gpu_environment_manifest.json` and `gpu_pip_freeze.txt` when retrieving results.

## Completion and retrieval (15 September 2026)

Remote status was COMPLETE, with 60/60 atomic job markers and a final completion
manifest. All 60 summary checksums passed remotely. The 27,986,286-byte result
archive was downloaded through JupyterLab and extracted separately under
`paper/aistats_followup/h200_results/` locally. Archive SHA-256:
`e6f982910b7f1940cb52754a2c256bab78a10edb039753c1917e69b91744a12c`.
All job artifacts passed their hashes locally. All 11,625 summary means were
recomputed from 2,325,000 per-draw evaluation rows, including the two degree controls.
No experiment restart or additional GPU allocation was needed.
