#!/bin/bash
# Submit the mesh-interpolation Phase 0 experiment to the SLURM cluster.
#
# Three dependency waves:
#   Wave 1  -- 10 GPU jobs, one per knee: fit the latent + cache its 4
#              reconstructed surfaces (experiments...fit_cache --key <knee>).
#   Wave 2  -- 8 GPU jobs, one per experiment config: run that config's slice
#              of the matrix (run_matrix --configs <cfg> --out-tag <cfg>).
#              Each waits (afterok) on ALL wave-1 jobs -- every pair spans all
#              knees, so the full cache must exist first.
#   Wave 3  -- 1 CPU job: merge the per-config shards into results.csv +
#              report.md. Waits (afterok) on all wave-2 jobs.
#
# Wave 2 is split by (config x NFE) so each job is small enough to finish
# inside the time limit; run_matrix checkpoints after every pair and resumes
# from its shard, so a timeout just needs a resubmit.
#
# Usage:
#   ./submit_phase0.sh --dry-run        # print the plan, submit nothing
#   ./submit_phase0.sh                  # fit + matrix + merge
#   ./submit_phase0.sh --matrix-only    # skip wave 1 (cache already built);
#                                       # also resumes partial matrix shards
#   ./submit_phase0.sh --matrix-only --configs=all,fix1_fix2_fix3
#                                       # restrict wave 2 to a config subset
#
# Monitor:  squeue -u $USER
# SLURM logs land in experiments/mesh_interpolation/slurm_outputs/

# Knees used for the matrix (ordered pairs = N*(N-1)). 5 -> 20 pairs; the
# manifest is KL-interleaved by run_matrix so 5 still spans KL0/1/2.
MAX_KNEES=5

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NSM_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SLURM_OUT="${SCRIPT_DIR}/slurm_outputs"
mkdir -p "$SLURM_OUT"
cd "$NSM_ROOT"  # so the `experiments` package imports below resolve

CONDA_SH="/dataNAS/people/aagatti/miniconda/etc/profile.d/conda.sh"
CONDA_ENV="comak"  # has nsosim + an editable install of this NSM repo

DRY_RUN=0
MATRIX_ONLY=0
CONFIG_FILTER=""   # comma-separated config subset for wave 2 (default: all)
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=1
    [[ "$arg" == "--matrix-only" ]] && MATRIX_ONLY=1
    [[ "$arg" == --configs=* ]] && CONFIG_FILTER="${arg#--configs=}"
done

# Knee keys, config names and the NFE grid -- pulled live from config.py /
# the manifest so this script never drifts from the experiment definition.
KEYS=$(python -c "import json; print(' '.join(r['key'] for r in json.load(open('${SCRIPT_DIR}/cache/manifest.json'))))")
CONFIGS=$(python -c "from experiments.mesh_interpolation.config import EXPERIMENT_CONFIGS as c; print(' '.join(c))")
NFE_GRID=$(python -c "from experiments.mesh_interpolation.config import NFE_GRID as n; print(' '.join(map(str, n)))")
if [[ -n "$CONFIG_FILTER" ]]; then
    CONFIGS="${CONFIG_FILTER//,/ }"
fi

echo "=================================================="
echo "Mesh-interpolation Phase 0"
echo "  Knees:   $(echo $KEYS | wc -w)   (wave 1, GPU; skipped if --matrix-only)"
echo "  Matrix:  $(echo $CONFIGS | wc -w) configs x $(echo $NFE_GRID | wc -w) NFE = "\
"$(( $(echo $CONFIGS | wc -w) * $(echo $NFE_GRID | wc -w) )) jobs (wave 2, GPU)"
echo "  + 1 merge job (wave 3, CPU)"
[[ $DRY_RUN -eq 1 ]] && echo "  *** DRY RUN -- nothing will be submitted ***"
echo "=================================================="

emit_job() {  # $1=path  $2=jobname  $3=sbatch-extra-lines  $4=command
    cat > "$1" <<SLURM
#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=24gb
$3
#SBATCH --output=${SLURM_OUT}/${2}-%j.out
#SBATCH --job-name=${2}
source ${CONDA_SH}
conda activate ${CONDA_ENV}
cd ${NSM_ROOT}
echo "${2} started \$(date) on \$(hostname)"
$4
echo "${2} finished \$(date) exit \$?"
SLURM
}

# ---- Wave 1: per-knee fitting (GPU) --------------------------------------
FIT_JOBS=()
if [[ $MATRIX_ONLY -eq 0 ]]; then
    for KEY in $KEYS; do
        JOB="fit_${KEY}"
        SH=$(mktemp "${SLURM_OUT}/${JOB}_XXXXXX.sh")
        emit_job "$SH" "$JOB" \
            $'#SBATCH --gres=gpu:1\n#SBATCH --time=0-04:00:00' \
            "python -m experiments.mesh_interpolation.fit_cache --key ${KEY}"
        if [[ $DRY_RUN -eq 1 ]]; then
            echo "[dry-run] wave1 ${JOB}"
            rm -f "$SH"
        else
            JID=$(sbatch "$SH" | awk '{print $NF}')
            FIT_JOBS+=("$JID")
            echo "wave1 ${JOB}: ${JID}"
            rm -f "$SH"
        fi
    done
else
    echo "wave1 skipped (--matrix-only); using existing cache/"
fi

# afterok dependency on every wave-1 job (empty when --matrix-only).
DEP=""
if [[ ${#FIT_JOBS[@]} -gt 0 ]]; then
    DEP="--dependency=afterok:$(IFS=:; echo "${FIT_JOBS[*]}")"
fi

# ---- Wave 2: per-(config x NFE) matrix slices (GPU) ----------------------
MATRIX_JOBS=()
for CFG in $CONFIGS; do
  for NFE in $NFE_GRID; do
    JOB="mtx_${CFG}__${NFE}"
    SH=$(mktemp "${SLURM_OUT}/${JOB}_XXXXXX.sh")
    emit_job "$SH" "$JOB" \
        $'#SBATCH --gres=gpu:1\n#SBATCH --time=1-00:00:00' \
        "python -m experiments.mesh_interpolation.run_matrix --configs ${CFG} --nfe ${NFE} --max-knees ${MAX_KNEES} --out-tag ${CFG}__${NFE}"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry-run] wave2 ${JOB} (afterok: all wave1)"
        rm -f "$SH"
    else
        JID=$(sbatch $DEP "$SH" | awk '{print $NF}')
        MATRIX_JOBS+=("$JID")
        echo "wave2 ${JOB}: ${JID} ${DEP}"
        rm -f "$SH"
    fi
  done
done

# ---- Wave 3: merge (CPU) -------------------------------------------------
JOB="phase0_merge"
SH=$(mktemp "${SLURM_OUT}/${JOB}_XXXXXX.sh")
emit_job "$SH" "$JOB" \
    $'#SBATCH --time=0-00:30:00' \
    "python -m experiments.mesh_interpolation.run_matrix --merge"
if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] wave3 ${JOB} (afterok: all wave2)"
    rm -f "$SH"
else
    MDEP="--dependency=afterok:$(IFS=:; echo "${MATRIX_JOBS[*]}")"
    JID=$(sbatch $MDEP "$SH" | awk '{print $NF}')
    echo "wave3 ${JOB}: ${JID} ${MDEP}"
    rm -f "$SH"
fi

echo "=================================================="
if [[ $DRY_RUN -eq 1 ]]; then
    echo "Dry run complete. Re-run without --dry-run to submit."
else
    echo "Submitted. Monitor: squeue -u \$USER"
    echo "Final report: ${SCRIPT_DIR}/report/report.md"
fi
echo "=================================================="
