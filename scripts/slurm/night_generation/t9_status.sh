#!/bin/bash
# Quick T9 status snapshot. Run via: ssh-mcp ssh_execute t9_status.sh
# (or any equivalent invocation that runs on the cluster).
#
# Reports SLURM status for jobs 1148 (S0) and 1149 (S5), plus the latest
# epoch / iter / loss line from each train.log if the run has started.

set -u
cd /srv/nfs/shared/gnmp/RaCFormer

echo "== squeue =="
squeue -j 1174,1175,1176,1178 -o "%.8i %.10P %.13j %.8T %.10M %.10L %R" 2>&1 | head

for s in S0:1171:racformer_train2k_day_research S5:1172:racformer_train2k_mixed_research S1:1176:racformer_train2k_simnight_research; do
  IFS=: read tag jid cfgname <<<"$s"
  echo
  echo "== $tag (job $jid) =="
  log=$(ls -dt outputs/$cfgname/*/* 2>/dev/null | head -1)
  if [ -z "${log:-}" ] || [ ! -f "$log/train.log" ]; then
    echo "  no train.log yet (run hasn't started or work_dir not created)"
    echo "  slurm logs:"
    ls -la research/night_gen_phase1/results/$tag/slurm_${jid}.* 2>/dev/null | head -2
    continue
  fi
  echo "  work_dir: $log"
  echo "  last Epoch line:"
  grep -E "Epoch \[" "$log/train.log" | tail -1 | sed 's/^/    /'
  echo "  last log line (any):"
  tail -1 "$log/train.log" | sed 's/^/    /'
  echo "  ckpts:"
  ls -la "$log"/*.pth 2>/dev/null | tail -5 | sed 's/^/    /'
done
