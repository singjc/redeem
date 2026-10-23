#!/usr/bin/env bash
#SBATCH --job-name=redeem-pt17-dev
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/research/juan/DIA/singj/foundation_model/redeem-pt17-dev-%j.out
#SBATCH --error=/nfs/research/juan/DIA/singj/foundation_model/redeem-pt17-dev-%j.err

ROOT='/nfs/research/juan/DIA/singj/foundation_model'
if ! module load singularityce/4.1.0; then
  echo 'REDEEM_PT2017_JOB=FAILED singularity module unavailable'
  /bin/false
elif ! python3 "$ROOT/scripts/redeem_pt2017_newdev_score_20261023.py" run; then
  echo 'REDEEM_PT2017_JOB=FAILED existing inputs and partial result retained'
  /bin/false
else
  echo 'REDEEM_PT2017_JOB=PASS'
fi
