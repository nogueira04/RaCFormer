#!/bin/bash
# Restore the /srv/nfs/shared bind mount (node3's layout) on a reimaged livenode.
# Idempotent: safe to re-run; only appends the fstab line if it is not already there.
set -euo pipefail
findmnt -n /mnt/nfs >/dev/null || mount /mnt/nfs
grep -qE '^/mnt/nfs[[:space:]]+/srv/nfs/shared[[:space:]]+none[[:space:]]+bind' /etc/fstab || \
  echo '/mnt/nfs    /srv/nfs/shared    none    bind    0   0' >> /etc/fstab
mkdir -p /srv/nfs/shared
findmnt -n /srv/nfs/shared >/dev/null || mount /srv/nfs/shared
ls /srv/nfs/shared/gnmp >/dev/null
echo "NODE_FIXED $(hostname)"
