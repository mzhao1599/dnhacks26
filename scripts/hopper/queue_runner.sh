#!/bin/bash
# Keep our GPU slots full from a task queue. Runs on the Hopper login node (only sbatch/squeue, no compute):
#   nohup bash scripts/hopper/queue_runner.sh > logs/queue_runner.log 2>&1 &
# Queue file: scripts/hopper/queue.txt, one task per line:   name | tier | command
#   tier = a100 | a80 (A100.80gb) | mig (3g.40gb) | mig1 (1g.10gb) | mig2 (2g.20gb) | cpu (-p normal, no GPU, c=32 default);
#   command runs inside `--wrap` after the standard
#   env prefix (see PREFIX below). All mig* tiers share the MAX_MIG cap and the osmesa prefix.
# Lines starting with # are ignored. A task is submitted once (recorded in logs/queue.done) when
# fewer than MAX_A100 / MAX_MIG of our jobs are running+pending on that tier. Append lines any time.
set -u
ROOT=/scratch/ezhao2/fleet-memory
REPO=$ROOT/dnhacks26
Q=$REPO/scripts/hopper/queue.txt
DONE=$ROOT/logs/queue.done
MAX_A100=${MAX_A100:-6}
MAX_A80=${MAX_A80:-12}      # QOS gpu allows a100.80gb=12 per user
MAX_MIG=${MAX_MIG:-3}
MAX_CPU=${MAX_CPU:-8}       # `cpu` tier: -p normal, no GPU (SmolVLA on CPU, osmesa)
POLL=${POLL:-60}
mkdir -p $ROOT/logs/slurm; touch "$DONE"
PREFIX_A100='source scripts/hopper/env.sh; export FM_POLICY_KWARGS="{\"n_action_steps\":10}" FM_LIBERO_PLUS=/scratch/ezhao2/fleet-memory/LIBERO-plus PYTHONPATH=$FM_REPO; cd $FM_REPO;'
PREFIX_MIG='export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa; '"$PREFIX_A100"' export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa;'
PREFIX_CPU='export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; '"$PREFIX_A100"' export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa CUDA_VISIBLE_DEVICES=;'

count() {  # our running+pending jobs on a tier, by job-name prefix q-a100- / q-mig-
  squeue -u ezhao2 -h -o "%j" | grep -c "^q-$1-"
}

while true; do
  na=$(count a100); nm=$(count mig); n8=$(count a80); nc=$(count cpu)
  while IFS= read -r line; do
    [ -z "$line" ] && continue; case "$line" in \#*) continue;; esac
    name=$(echo "$line" | cut -d'|' -f1 | xargs); tierf=$(echo "$line" | cut -d'|' -f2 | xargs); cmd=$(echo "$line" | cut -d'|' -f3- | sed 's/^ *//')
    grep -qx "$name" "$DONE" && continue
    # tier field: "a100" | "mig*", optionally followed by "after=<name>" (slurm afterany dependency on that
    # task's job, looked up by job name while it is still queued/running; no-op once finished) and/or "c=<n>" (CPUs, default 8).
    tier=${tierf%% *}; dep=""; cpus=8
    case "$tierf" in *c=*) cpus=$(echo "$tierf" | sed -n 's/.*c=\([0-9]*\).*/\1/p'); [ -n "$cpus" ] || cpus=8;; esac
    case "$tierf" in *after=*)
      after=${tierf##*after=}
      djid=$(squeue -u ezhao2 -h -o "%i %j" | awk -v a="q-a100-$after" -v m="q-mig-$after" -v r="$after" '$2==a||$2==m||$2==r{print $1; exit}')
      [ -n "$djid" ] && dep="--dependency=afterany:$djid";;
    esac
    if [ "$tier" = "a80" ] && [ "$n8" -lt "$MAX_A80" ]; then
      jid=$(sbatch --parsable $dep -p gpuq -q gpu --gres=gpu:A100.80gb:1 -c $cpus --mem=48G -t 04:00:00 -J "q-a80-$name" -o "$ROOT/logs/slurm/q_${name}_%j.out" --wrap "$PREFIX_A100 $cmd; echo QTASK_DONE $name") && { echo "$name" >> "$DONE"; echo "$(date +%H:%M) submitted $name ($jid) on a80 ${dep}"; n8=$((n8+1)); }
    elif [ "$tier" = "cpu" ] && [ "$nc" -lt "$MAX_CPU" ]; then
      [ "$cpus" -gt 8 ] || cpus=32
      jid=$(sbatch --parsable $dep -p normal -q normal -c $cpus --mem=96G -t 04:00:00 -J "q-cpu-$name" -o "$ROOT/logs/slurm/q_${name}_%j.out" --wrap "$PREFIX_CPU $cmd; echo QTASK_DONE $name") && { echo "$name" >> "$DONE"; echo "$(date +%H:%M) submitted $name ($jid) on cpu (c=$cpus) ${dep}"; nc=$((nc+1)); }
    elif [ "$tier" = "a100" ] && [ "$na" -lt "$MAX_A100" ]; then
      jid=$(sbatch --parsable $dep -p gpuq -q gpu --gres=gpu:A100.40gb:1 -c $cpus --mem=48G -t 04:00:00 -J "q-a100-$name" -o "$ROOT/logs/slurm/q_${name}_%j.out" --wrap "$PREFIX_A100 $cmd; echo QTASK_DONE $name") && { echo "$name" >> "$DONE"; echo "$(date +%H:%M) submitted $name ($jid) on a100 ${dep}"; na=$((na+1)); }
    elif { [ "$tier" = "mig" ] || [ "$tier" = "mig1" ] || [ "$tier" = "mig2" ]; } && [ "$nm" -lt "$MAX_MIG" ]; then
      case "$tier" in mig1) gres=gpu:1g.10gb:1;; mig2) gres=gpu:2g.20gb:1;; *) gres=gpu:3g.40gb:1;; esac
      jid=$(sbatch --parsable $dep -p gpuq -q gpu --gres=$gres -c $cpus --mem=48G -t 04:00:00 -J "q-mig-$name" -o "$ROOT/logs/slurm/q_${name}_%j.out" --wrap "$PREFIX_MIG $cmd; echo QTASK_DONE $name") && { echo "$name" >> "$DONE"; echo "$(date +%H:%M) submitted $name ($jid) on $tier ($gres) ${dep}"; nm=$((nm+1)); }
    fi
  done < "$Q"
  sleep "$POLL"
done
