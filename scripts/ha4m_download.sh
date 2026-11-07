#!/usr/bin/env bash
# HA4M download in order, one stream (Week 10). Run it in the tmux window "dl":
#   tmux new-window -d -n dl ~/AssemblyVision/scripts/ha4m_download.sh
# 1. annotations of every video (done videos are skipped, failed ones retried, up to 3 passes)
# 2. then the RGB of the 2 pilot videos (approved); the 24 video subset is NOT started here.
# Every pass resumes per video; progress and HTTP 429 waits go to logs/ha4m_download.log.
set -u
source ~/AssemblyVision/av.sh >/dev/null 2>&1
cd ~/AssemblyVision/code
LOG=~/AssemblyVision/logs/ha4m_download.log
note() { echo "$(date '+%F %T') INFO ha4m_download.sh: $*" | tee -a "$LOG"; }
X="python -m assemblyvision.tools.fetch_dataset ha4m"

note "start: annotations, then the RGB pilot (one stream)"
ok=0
for pass in 1 2 3; do
    $X annotations --videos all
    code=$?
    if [ $code -eq 0 ]; then ok=1; break; fi
    if [ $code -eq 2 ]; then note "annotations refused (exit 2), stopping"; exit 2; fi
    note "annotations pass $pass ended with failures; next pass in 10 min"
    sleep 600
done
if [ $ok -ne 1 ]; then note "annotations still incomplete after 3 passes; RGB not started"; exit 1; fi
note "annotations complete; RGB pilot next"
$X rgb --videos IDU001V001,IDU023V001 --yes
code=$?
note "RGB pilot ended with exit $code; the 24 video subset waits for approval"
exit $code
