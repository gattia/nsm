# Mesh-interpolation Phase 0 experiment

Runner for Phase 0 of the `NSM_MESH_INTERPOLATION_IMPROVEMENTS` plan: build a
correspondence-quality baseline for `NSM.mesh.interpolate.interpolate_points`
and score the six numerical fixes across the experiment matrix.

## Pipeline

Three sequential steps. Step 1 needs no GPU; steps 2-3 do (see below).

```bash
# Step 1 -- select 10 KL-stratified pilot knees, write cache/manifest.json.
#           Needs only the demographics CSV. CPU-only.
python -m experiments.mesh_interpolation.subjects

# Step 2 -- fit a latent per knee and cache its 4 reconstructed surfaces.
#           REQUIRES A CUDA GPU (nsosim.load_model moves the model to CUDA).
#           Resumable -- already-cached knees are skipped.
python -m experiments.mesh_interpolation.fit_cache

# Step 3 -- run the config x NFE x pair x surface matrix, write report/.
python -m experiments.mesh_interpolation.run_matrix
```

`run_matrix` accepts flags to subset the (large) matrix -- start small:

```bash
python -m experiments.mesh_interpolation.run_matrix \
    --configs baseline,fix1_fix2,all --nfe 50,100 --max-pairs 6
```

## SLURM submission (recommended)

`submit_phase0.sh` runs the whole pipeline as 19 parallel cluster jobs in
three dependency waves: 10 GPU fit jobs, then 8 GPU matrix jobs (one per
config, `afterok` on all fits), then 1 CPU merge job. It uses the `comak`
conda env, which has `nsosim` and an editable install of this NSM repo.

```bash
./experiments/mesh_interpolation/submit_phase0.sh --dry-run   # preview
./experiments/mesh_interpolation/submit_phase0.sh             # submit
squeue -u $USER                                               # monitor
```

Each matrix job writes a `report/results_<config>.csv` shard; the merge job
concatenates them (`run_matrix --merge`) into `report/results.csv` +
`report/report.md`.

## Smoke test (no GPU, no data)

Verifies the harness end-to-end (every fix config -> scoring -> report) against
an analytic sphere-SDF decoder:

```bash
python -m experiments.mesh_interpolation.smoke_test
```

## GPU requirement

`nsosim.utils.load_model` hard-codes `model.cuda()`, so **steps 2 and 3 must run
on a node with a CUDA GPU**. `config.load_nsm_model` (used by `run_matrix`) is
device-aware and will fall back to CPU, but `fit_cache` goes through nsosim and
cannot. The full matrix is also large -- 8 configs x 5 NFE x 90 ordered pairs x
4 surfaces x 2 warps -- and is intended for a GPU.

## Files

| File | Role |
|------|------|
| `config.py` | Paths, the model spec, the experiment matrix, shared helpers. |
| `subjects.py` | Step 1 -- KL-stratified subject selection -> `cache/manifest.json`. |
| `fit_cache.py` | Step 2 -- per-knee latent fitting + marching-cubes caching. |
| `run_matrix.py` | Step 3 -- the experiment matrix, scoring, and report. |
| `smoke_test.py` | GPU-free end-to-end harness check. |

## Outputs

- `cache/manifest.json` -- selected knees + mesh paths.
- `cache/<key>_latent.npy`, `cache/<key>_<surface>.vtk` -- fitted latents and
  reconstructed surfaces.
- `report/results.csv`, `report/results.json` -- long-format per-cell scores.
- `report/report.md` -- aggregated per-surface / NFE-sensitivity summary.
